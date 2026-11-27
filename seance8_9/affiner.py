import json
import random
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MOTEUR, SORTIE = sys.argv[1], sys.argv[2]

# R?glages de d?part pour les 300 exemples de paires.json.
EPOCHES = 3
BATCH_EFFECTIVE = 4
TAUX_APPRENTISSAGE = 1e-4

tokeniseur = AutoTokenizer.from_pretrained(MOTEUR)
moteur = AutoModelForCausalLM.from_pretrained(MOTEUR, torch_dtype=torch.bfloat16, device_map="cuda")
moteur.gradient_checkpointing_enable()
moteur.enable_input_require_grads()

moteur = get_peft_model(moteur, LoraConfig(r=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")) # configuration prête , on a pas encore entraîné l'adaptateur.
entraines = [p for p in moteur.parameters() if p.requires_grad]  # stocker les paramètres entraînables de l'adaptateur dans la liste entraines, pour les entraîner plus tard. 
                                                                 # Les autres paramètres du moteur sont gelés, ils ne seront pas entraînés.
for p in entraines:
    p.data = p.data.float()   # .data pour accéder au valeurs numérique de tensor, les changer en float32 (x32) pour l'entraînement
                              # effectuer calcules sur CPU 
print("paramètres entraînés", sum(p.numel() for p in entraines))


def exemple(paire):
    demande = tokeniseur.apply_chat_template(
        [{"role": "user", "content": paire["demande"]}], tokenize=False, add_generation_prompt=True)
    ids_demande = tokeniseur(demande)["input_ids"]   # je serais besoin de cette info lors de l'application de masque
    ids = tokeniseur(demande + paire["reponse"] + "<|im_end|>")["input_ids"]
    cibles = [-100] * len(ids_demande) + ids[len(ids_demande):]
    return torch.tensor([ids]).cuda(), torch.tensor([cibles]).cuda()


paires = json.load(open("paires.json", encoding="utf-8"))
adam = torch.optim.AdamW(entraines, lr=TAUX_APPRENTISSAGE)
random.seed(0)
for epoque in range(EPOCHES):
    random.shuffle(paires)
    pertes = []
    for debut in range(0, len(paires), BATCH_EFFECTIVE):
        groupe = paires[debut:debut + BATCH_EFFECTIVE]
        adam.zero_grad(set_to_none=True)
        for paire in groupe:
            ids, cibles = exemple(paire)
            perte = moteur(input_ids=ids, labels=cibles).loss
            (perte / len(groupe)).backward()
            pertes.append(perte.item())
        torch.nn.utils.clip_grad_norm_(entraines, 1.0)
        adam.step()
    print(f"?poque {epoque + 1}, perte moyenne {sum(pertes) / len(pertes):.3f}")

for p in entraines:
    p.data = p.data.to(torch.float16)
moteur.save_pretrained(SORTIE)
print("adaptateur enregistré dans le dossier", SORTIE)