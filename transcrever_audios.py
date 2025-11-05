#!/usr/bin/env python3
"""
Script para transcrever arquivos de áudio usando Faster-Whisper
"""
from faster_whisper import WhisperModel
import os
import json
from pathlib import Path
import time

def transcrever_audio(arquivo_audio, model):
    """
    Transcreve um arquivo de áudio usando Faster-Whisper

    Args:
        arquivo_audio: Caminho para o arquivo de áudio
        model: Modelo WhisperModel já carregado

    Returns:
        Dict com a transcrição e metadados
    """
    print(f"Transcrevendo {arquivo_audio}...")
    inicio = time.time()

    segments, info = model.transcribe(str(arquivo_audio), language="pt", beam_size=5)

    # Juntar todos os segmentos
    texto_completo = ""
    segmentos_lista = []

    for segment in segments:
        texto_completo += segment.text + " "
        segmentos_lista.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })

    tempo_decorrido = time.time() - inicio

    return {
        "text": texto_completo.strip(),
        "segments": segmentos_lista,
        "language": info.language,
        "duration": info.duration,
        "tempo_processamento": tempo_decorrido
    }

def main():
    # Diretório com os áudios
    curso_dir = Path("/home/user/generico/curso")

    # Diretório para salvar transcrições
    output_dir = Path("/home/user/generico/transcricoes")
    output_dir.mkdir(exist_ok=True)

    # Listar todos os arquivos MP3
    arquivos_audio = sorted(curso_dir.glob("part_*.mp3")) + sorted(curso_dir.glob("class01_*.mp3"))

    print(f"Encontrados {len(arquivos_audio)} arquivos de áudio")
    print("Carregando modelo Faster-Whisper 'base'...")

    # Carregar modelo uma vez só (CPU, float32)
    model = WhisperModel("base", device="cpu", compute_type="int8")

    print("Modelo carregado!")
    print("=" * 60)

    # Transcrever cada arquivo
    for i, arquivo in enumerate(arquivos_audio):
        # Pular se já existe
        txt_file = output_dir / f"{arquivo.stem}_transcricao.txt"
        if txt_file.exists():
            print(f"\n[{i+1}/{len(arquivos_audio)}] PULANDO {arquivo.name} (já processado)")
            continue

        print(f"\n[{i+1}/{len(arquivos_audio)}] Processando {arquivo.name}")
        print("-" * 60)

        try:
            # Transcrever
            resultado = transcrever_audio(arquivo, model)

            # Salvar transcrição em texto
            txt_file = output_dir / f"{arquivo.stem}_transcricao.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(resultado["text"])

            # Salvar metadados em JSON
            json_file = output_dir / f"{arquivo.stem}_transcricao.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(resultado, f, ensure_ascii=False, indent=2)

            print(f"✓ Transcrição concluída em {resultado['tempo_processamento']:.1f}s")
            print(f"  Duração do áudio: {resultado['duration']:.1f}s")
            print(f"  Arquivos salvos:")
            print(f"    - {txt_file}")
            print(f"    - {json_file}")
            print(f"\n  Preview: {resultado['text'][:150]}...")

        except Exception as e:
            print(f"✗ Erro ao transcrever {arquivo.name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Transcrição concluída!")
    print(f"Arquivos salvos em: {output_dir}")

if __name__ == "__main__":
    main()
