import json
import math
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

# AMELIORATION : hyperparamètres regroupés en haut, faciles à modifier
EPOQUES = 6
ACCUM = 8  # accumulation de gradient : on met à jour les poids tous les 8 exemples

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()
# AMELIORATION : le cache est incompatible avec le gradient checkpointing (évite un warning)
moteur.config.use_cache = False

# AMELIORATION : lora_alpha=32 (avant : valeur par défaut 8). Le scaling LoRA = alpha/r,
# donc avec alpha=8 et r=16 l'adaptateur était trop faible (0.5). Maintenant il vaut 2.
# AMELIORATION : on adapte aussi k_proj, o_proj et les couches MLP (gate/up/down),
# pas seulement q_proj et v_proj : plus de capacité pour apprendre le style/format.
# AMELIORATION : lora_dropout=0.05 pour limiter le sur-apprentissage avec peu de données.
moteur = get_peft_model(moteur, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
))

# AMELIORATION : get_peft_model laisse le modèle en mode eval. Sans .train(), le dropout LoRA
# et le gradient checkpointing ne sont pas actifs.
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
    # AMELIORATION : "\n" après <|im_end|> pour coller exactement au template chat de Qwen
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>\n")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


paires = json.load(open("paires.json", encoding="utf-8"))
random.seed(0)
torch.manual_seed(0)  # AMELIORATION : résultats reproductibles

# AMELIORATION : jeu de validation (10 %). Sans lui, impossible de savoir si on sous-apprend
# ou si on sur-apprend : la perte d'entraînement seule ne suffit pas.
random.shuffle(paires)
taille_val = max(1, int(0.1 * len(paires)))
val, train = paires[:taille_val], paires[taille_val:]
print(f"{len(train)} paires d'entraînement, {len(val)} paires de validation")


def perte_validation():
    moteur.eval()  # pas de dropout pendant la mesure
    total = 0.0
    with torch.no_grad():  # pas de gradient : plus rapide et moins de mémoire
        for paire in val:
            ids, cibles = exemple(paire)
            total += moteur(input_ids=ids, labels=cibles).loss.item()
    moteur.train()  # on remet en mode entraînement
    return total / len(val)


# AMELIORATION : weight_decay + scheduler cosinus avec warmup (10 %).
# Avant : lr constant et un step par exemple, donc entraînement très bruité.
adam = torch.optim.AdamW(entraines, lr=2e-4, weight_decay=0.01)
total_pas = math.ceil(len(train) / ACCUM) * EPOQUES
planif = get_cosine_schedule_with_warmup(adam, int(0.1 * total_pas), total_pas)

meilleure_val = float("inf")

for epoque in range(EPOQUES):
    random.shuffle(train)
    pertes = []
    for i, paire in enumerate(train):
        ids, cibles = exemple(paire)
        perte = moteur(input_ids=ids, labels=cibles).loss
        # AMELIORATION : on divise par ACCUM pour que la somme des gradients soit une moyenne
        (perte / ACCUM).backward()
        pertes.append(perte.item())#TONSEUR avec un seul element donc je recupere la valeur avec item() et je l'ajoute a la liste des pertes pour calculer la perte moyenne a la fin de l'epoque

        # AMELIORATION : mise à jour tous les ACCUM exemples (et à la fin pour le dernier lot incomplet)
        if (i + 1) % ACCUM == 0 or (i + 1) == len(train):
            # AMELIORATION : clipping pour éviter les gradients qui explosent
            torch.nn.utils.clip_grad_norm_(entraines, 1.0)
            adam.step()
            planif.step()
            adam.zero_grad()

    pv = perte_validation()
    print(f"époque {epoque + 1}, perte train {sum(pertes) / len(pertes):.3f}, perte val {pv:.3f}")

    # AMELIORATION : on sauvegarde le MEILLEUR adaptateur (perte val minimale), pas forcément
    # celui de la dernière époque, qui peut avoir sur-appris.
    # AMELIORATION : sauvegarde en float32, sans conversion en float16 (pas de perte de précision,
    # et le fichier reste petit car LoRA ne contient que les matrices A et B).
    if pv < meilleure_val:
        meilleure_val = pv
        moteur.save_pretrained(SORTIE)
        print("  -> meilleur adaptateur enregistré dans le dossier", SORTIE)

print(f"terminé, meilleure perte de validation : {meilleure_val:.3f}")
#donc les matrice A et B doivent etre crier