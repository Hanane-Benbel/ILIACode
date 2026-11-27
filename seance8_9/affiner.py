import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()

# CHANGÉ : target_modules étendu (avant seulement q_proj et v_proj) pour plus de capacité d'apprentissage
# AJOUTÉ : lora_alpha=32 (par défaut 8, donc adaptateur trop faible : scaling = alpha/r)
# AJOUTÉ : lora_dropout=0.05 pour limiter le sur-apprentissage avec peu de données
moteur = get_peft_model(moteur, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"))
# AJOUTÉ : sans .train() le modèle reste en mode eval : le dropout et le gradient checkpointing ne sont pas actifs
moteur.train()
entraines = [p for p in moteur.parameters() if p.requires_grad]
for p in entraines:
    p.data = p.data.float()
print("paramètres entraînés", sum(p.numel() for p in entraines))

#cette fonction contient  
def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]
    # CHANGÉ : "<|im_end|>" -> "<|im_end|>\n" pour coller exactement au template de Qwen
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>\n")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


paires = json.load(open("paires.json", encoding="utf-8"))
# CHANGÉ : lr 2e-4 -> 1e-4 (batch de 1 + plus de couches adaptées = lr plus petit pour rester stable)
# AJOUTÉ : weight_decay=0.01 (régularisation)
adam = torch.optim.AdamW(entraines, lr=1e-4, weight_decay=0.01)
random.seed(0)
# CHANGÉ : 6 -> 12 époques (avec seulement 20 paires, 6 époques = 120 pas, trop peu pour apprendre un style)
for epoque in range(12):
    random.shuffle(paires)
    pertes = []
    for paire in paires:
        ids, cibles = exemple(paire)
        perte = moteur(input_ids=ids, labels=cibles).loss
        perte.backward()
        # AJOUTÉ : clipping du gradient pour éviter les pas qui explosent
        torch.nn.utils.clip_grad_norm_(entraines, 1.0)
        adam.step()
        adam.zero_grad()
        pertes.append(perte.item())#TONSEUR avec un seul element donc je recupere la valeur avec item() et je l'ajoute a la liste des pertes pour calculer la perte moyenne a la fin de l'epoque
    print(f"époque {epoque + 1}, perte moyenne {sum(pertes) / len(pertes):.3f}")#on a 20 paires dans chaque époque donc on va calculer la perte moyenne sur les 20 paires et l'afficher a la fin de chaque époque

for p in entraines:#tonseur par tonseur je convertis les poids en float16 pour economiser de la memoire et je sauvegarde le modele dans le dossier SORTIE
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)
#donc les matrice A et B doivent etre crier