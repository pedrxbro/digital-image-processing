# Projeto M2 - Processamento Digital de Imagens

## Objetivo

Desenvolver um pipeline clássico de processamento digital de imagens para segmentação automática de células em imagens microscópicas.

## Sprint 1 - Base do projeto e dataset

Nesta sprint foram criadas a estrutura inicial do projeto, o notebook principal, o carregamento das imagens selecionadas, a visualização inicial, os histogramas e a validação das máscaras de referência.

As imagens de amostra ficam em `data/samples/images`. As máscaras correspondentes devem ficar em `data/samples/masks`, preferencialmente com o mesmo nome-base das imagens.

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
- `notebooks`: contém o notebook principal da Sprint 1.
- `src`: contém funções reutilizáveis para carregamento, visualização e análise.
- `outputs`: armazena figuras e relatórios gerados durante os experimentos.

## Observação sobre máscaras

O notebook valida automaticamente se cada imagem possui uma máscara correspondente, se as dimensões são compatíveis e se a máscara é binária. Caso alguma máscara ainda não exista, a pendência será registrada na seção final do notebook.
