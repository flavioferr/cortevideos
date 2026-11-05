# CorteVideos

Transforma vídeos longos (podcasts, lives, aulas) em cortes verticais 9:16 com legenda sincronizada.
O **Claude** escolhe os melhores momentos, edita cada corte e ajusta tudo por chat.

Inspirado na arquitetura do [SupoClip](https://github.com/FujiwaraChoki/supoclip), mas com implementação própria e enxuta.

## Como funciona

```
URL ou upload
  → download (yt-dlp)
  → transcrição local com tempo de cada palavra (faster-whisper)
  → Claude: escolhe os cortes, dá nota de viralidade, escreve título-gancho, título, descrição e hashtags
  → Claude: refina cada corte (início/fim limpos, corrige a transcrição, remove vícios e gaguejadas,
            escolhe palavras de destaque e emojis)
  → remoção automática de pausas
  → câmera virtual 9:16 que acompanha o rosto (OpenCV) + legenda animada (libass) → MP4
     (ou layout gameplay: jogo em cima e a webcam do streamer embaixo, detectada automaticamente)
  → Editor por chat: "legenda verde e maior", "corta a primeira frase", "fundo desfocado"…
     o Claude usa ferramentas para alterar o clipe e ele é renderizado de novo
```

O Claude não recebe o áudio, por isso a transcrição é feita localmente pelo Whisper e o Claude trabalha em cima do texto com os tempos de cada palavra.

## Instalação (Windows)

Requer Python 3.11+. O FFmpeg já vem embutido (pacote `imageio-ffmpeg`).

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
copy .env.example .env
.\iniciar.bat
```

### Usando sua assinatura do Claude (padrão)

Com `CLAUDE_BACKEND=claude-code` o app chama o [Claude Code](https://claude.com/claude-code) instalado no PC
(`claude -p`), que usa o login da sua assinatura Pro/Max. Não há custo extra, mas o uso conta na cota da
assinatura. Basta ter o Claude Code instalado e logado (rode `claude` uma vez no terminal).

Para poupar a cota: `CLAUDE_MODEL=claude-sonnet-5-5` e/ou `CLAUDE_EFFORT=medium` no `.env`.

### Usando a API

Com `CLAUDE_BACKEND=api` e `ANTHROPIC_API_KEY` preenchida, o app usa a API da Anthropic (cobrança por uso).
Nesse modo o editor por chat usa ferramentas nativas e há fallback automático se uma requisição for recusada.

Abra http://127.0.0.1:8000.

Na primeira transcrição o modelo do Whisper é baixado (~460 MB no `small`).
Para ter mais precisão use `WHISPER_MODEL=medium` ou `large-v3` (mais lento em CPU).

## Onde o Claude entra

| Etapa | Arquivo | O que o Claude faz |
|---|---|---|
| Seleção | `backend/app/claude_ai.py` → `select_clips` | Lê a transcrição inteira e devolve os cortes em JSON (saída estruturada) |
| Edição | `claude_ai.py` → `edit_clip` | Ajusta os limites, corrige palavras, marca cortes, destaques e emojis |
| Chat | `claude_ai.py` → `chat_edit` | Loop de ferramentas (`set_style`, `cut_words`, `set_bounds`, `edit_words`…) |

Modelo padrão: `claude-opus-5-5`. Nos dois modos a resposta do Claude é validada por um JSON Schema.

## Música de fundo

Importe uma playlist do YouTube na seção **Biblioteca de músicas**. Cada faixa é:

1. **Analisada por algoritmos de MIR** (librosa): BPM por beat tracking (Ellis, 2007), tom e modo maior/menor
   por perfis de Krumhansl-Schmuckler, energia RMS, brilho (centroide espectral) e taxa de ataques.
   Esses descritores posicionam a faixa no **modelo circumplexo de Russell** (energia × valência).
2. **Classificada pelo Claude**, que reconhece a música pelo nome: climas, gênero, se tem voz cantada,
   e corrige a estimativa do algoritmo quando ela erra (ex.: BPM detectado pela metade).

Para cada corte, o Claude descreve o clima ideal; um algoritmo dá nota a todas as faixas (distância no
plano energia × valência, climas em comum, penalidade para voz cantada e para repetição) e o Claude escolhe
entre as 6 melhores. A música entra num trecho com energia adequada ao clima (refrão/drop para conteúdo
agitado), alinhada à batida, com loudness normalizada, fades e **ducking** (compressor sidechain que abaixa
a música enquanto alguém fala). No chat dá para pedir "troca a música", "abaixa a música" etc.

> Atenção a direitos autorais: músicas comerciais em vídeos postados podem ser silenciadas ou gerar
> reivindicação (principalmente no YouTube). Para postar sem risco, prefira trilhas livres de direitos.

## Layout gameplay (jogo em cima, webcam embaixo)

Para lives e gameplays com a webcam num canto da tela. A webcam é encontrada sozinha: num trecho do
vídeo, as bordas que ficam paradas (moldura e quarto do streamer) se destacam do jogo, que muda o tempo
todo; entre essas regiões, fica a que tem um rosto, ajustada às linhas retas da moldura. A legenda vai
na divisa entre o jogo e a webcam. Se nenhuma webcam for encontrada (vídeo só de câmera, podcast), o
corte usa o modo "seguir rosto".

## Estrutura

```
backend/app/
  main.py        API (FastAPI) e frontend
  pipeline.py    orquestra o processamento
  claude_ai.py   prompts, schemas e ferramentas do Claude
  transcribe.py  faster-whisper
  download.py    yt-dlp
  media.py       detecção de rosto e render com FFmpeg (incluindo a mixagem da música)
  music.py       biblioteca de músicas, análise com librosa e algoritmo de compatibilidade
  captions.py    legendas .ass palavra a palavra
  store.py       jobs e clipes salvos em data/
frontend/        HTML/CSS/JS sem build
fonts/           coloque .ttf aqui para usar nas legendas
```

## Limitações conhecidas

- Emojis na legenda saem monocromáticos (limite do libass).
- A câmera segue o rosto principal (o maior/mais próximo), não necessariamente quem está falando: com duas pessoas
  lado a lado no mesmo plano ela não alterna sozinha entre elas.
- Se o YouTube bloquear o download, exporte os cookies do navegador e defina `YTDLP_COOKIES_FILE`.
- Sem login/multiusuário: é feito para rodar localmente.
