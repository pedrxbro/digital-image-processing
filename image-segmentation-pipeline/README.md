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
