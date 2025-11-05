#!/usr/bin/env python3
"""Script para transcrever UM arquivo específico"""
import sys
from faster_whisper import WhisperModel
from pathlib import Path
import json
import time

if len(sys.argv) < 2:
    print("Uso: python3 transcrever_um.py <arquivo.mp3>")
    sys.exit(1)

arquivo_audio = sys.argv[1]
output_dir = Path("/home/user/generico/transcricoes")
output_dir.mkdir(exist_ok=True)

arquivo_path = Path(arquivo_audio)
nome = arquivo_path.stem

print(f"Carregando modelo...")
model = WhisperModel("base", device="cpu", compute_type="int8")

print(f"Transcrevendo {arquivo_path.name}...")
inicio = time.time()

segments, info = model.transcribe(str(arquivo_path), language="pt", beam_size=5)

texto_completo = ""
segmentos_lista = []

for segment in segments:
    texto_completo += segment.text + " "
    segmentos_lista.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })

tempo = time.time() - inicio

resultado = {
    "text": texto_completo.strip(),
    "segments": segmentos_lista,
    "language": info.language,
    "duration": info.duration,
    "tempo_processamento": tempo
}

# Salvar
txt_file = output_dir / f"{nome}_transcricao.txt"
json_file = output_dir / f"{nome}_transcricao.json"

with open(txt_file, "w", encoding="utf-8") as f:
    f.write(resultado["text"])

with open(json_file, "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

print(f"✓ Concluído em {tempo:.1f}s")
print(f"  TXT: {txt_file}")
print(f"  JSON: {json_file}")
