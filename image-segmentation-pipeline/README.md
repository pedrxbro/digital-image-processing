# Projeto M2 - Processamento Digital de Imagens

## Objetivo

Desenvolver um pipeline clássico de processamento digital de imagens para segmentação automática de células em imagens microscópicas.

## Organização do projeto

O projeto contém um notebook principal com o fluxo de trabalho de PDI: carregamento das imagens, visualização inicial, análise por histogramas, pré-processamento e filtragem no domínio da frequência.

As imagens de amostra ficam em `data/samples/images`. As máscaras correspondentes devem ficar em `data/samples/masks`, preferencialmente com o mesmo nome-base das imagens.

## Pré-processamento e Frequência

Foram implementadas técnicas de preparação das imagens para segmentação, incluindo conversão entre espaços de cor, normalização, equalização de histograma, Transformada de Fourier e filtros no domínio da frequência.

As implementações de pré-processamento e frequência ficam em `src/preprocessing.py` e `src/frequency.py`. As rotinas seguem as práticas trabalhadas nos materiais oficiais da disciplina para equalização por histograma/PDF/CDF e filtragem em frequência com FFT, máscaras circulares e reconstrução por transformada inversa. A OpenCV permanece apenas como apoio para leitura dos arquivos de imagem; as conversões de cor, equalização e filtros foram implementados com NumPy.

### Técnicas avaliadas

- Conversão RGB, HSV, LAB e escala de cinza.
- Normalização de intensidade.
- Equalização de histograma.
- FFT.
- Filtro passa-baixa.
- Filtro passa-alta.
- Comparação visual dos resultados.

### Observações iniciais

Nas amostras iniciais, os canais de saturação do HSV e os canais cromáticos do LAB tendem a destacar melhor as células coradas em relação ao fundo claro. O canal L do LAB ajuda a analisar variações de luminosidade, mas a equalização deve ser usada com cautela porque pode aumentar ruídos, halos e artefatos do preparo. Os filtros passa-baixa são úteis para suavização antes da segmentação, enquanto os passa-alta destacam bordas, mas também podem amplificar detalhes indesejados.

## Segmentação de Células por Superpixels

Nesta etapa, os superpixels gerados pelo SLIC foram usados como unidades de decisão para construir uma máscara binária inicial das células.

Foram calculados atributos por superpixel, incluindo médias dos canais HSV e LAB, saturação média, luminosidade média e intensidade média. Em seguida, foram testadas estratégias de limiarização global e Otsu por superpixel implementado From Scratch.

O objetivo foi classificar cada superpixel como célula ou fundo e gerar uma máscara binária inicial para posterior refinamento com morfologia matemática.

## Como executar

1. Criar ambiente virtual:

```bash
python -m venv .venv
```

2. Ativar ambiente:

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

3. Instalar dependências:

```bash
pip install -r requirements.txt
```

4. Abrir o notebook:

```bash
jupyter notebook notebooks/main.ipynb
```

## Estrutura do projeto

- `data`: armazena dados brutos, dados processados e amostras usadas nos notebooks.
- `notebooks`: contém o notebook principal do projeto.
- `src`: contém funções reutilizáveis para carregamento, visualização e análise.
- `outputs`: armazena figuras e relatórios gerados durante os experimentos.

## Pipeline configuravel para hemacias

A orquestracao reutilizavel esta em `src/pipeline.py`. Ela preserva a imagem RGB
para visualizacao, mantem grayscale explicitamente e permite selecionar canais
2D de RGB, HSV, LAB ou GRAY para segmentacao. O notebook `notebooks/main.ipynb`
apenas configura, executa e explica o experimento.

O realce de bordas usa Sobel from scratch em `src/edge_enhancement.py`. O
preenchimento conservador de pequenos contornos e o Watershed manual ficam em
`src/morphology.py`. A versao final usa regras sobre atributos dos superpixels
para recuperar hemacias visiveis e palidas e excluir regioes azuladas.

Relatorios detalhados, mascaras, overlays e mapas intermediarios sao gravados
em `outputs/pipeline_final`. Cada imagem recebe sua propria pasta com
`config.json` e `execution_report.json`. O lote tambem gera
`batch_summary.csv` e `batch_summary.json`.

### Resultado de referencia

A configuracao final usa `LAB.A`, filtro passa-baixa com raio 35, SLIC com 450
superpixels, abertura 5x5, fechamento 3x3 e Watershed com 12 erosoes para gerar
marcadores. Na execucao validada em 31 de maio de 2026:

| Imagem | Componentes antes do Watershed | Hemacias estimadas |
|---|---:|---:|
| `BloodImage_00340.jpg` | 18 | 27 |
| `BloodImage_00367.jpg` | 8 | 18 |
| `BloodImage_00368.jpg` | 5 | 27 |

As contagens sao estimativas: sem mascaras de referencia, os overlays devem ser
inspecionados e Dice/IoU nao podem ser reportados de forma honesta.

Exemplos de selecao de entrada:

```python
{"input_strategy": {"channel_source": "GRAY", "selected_channel": "GRAY"}}
{"input_strategy": {"channel_source": "HSV", "selected_channel": "S"}}
{"input_strategy": {"channel_source": "LAB", "selected_channel": "A"}}
{"input_strategy": {"channel_source": "LAB", "selected_channel": "B"}}
```
