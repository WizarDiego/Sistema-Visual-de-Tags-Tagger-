# Sistema Visual de Tags & Batch Editor (Frutiger Aero)

Um aplicativo desktop desenvolvido em Python (PyQt6) com a bela e nostálgica estética **Frutiger Aero**. Ele foi projetado para facilitar o processamento rápido em lote de imagens e o gerenciamento visual de *datasets* (textos e tags) para o treinamento de IAs e LoRAs.

## 🚀 Funcionalidades Atuais

1. **Visualizador e Anotador (Tagger Visual):**
   - Interface moderna e responsiva onde as tags (prompts) são representadas por botões de vidro (glassmorphism).
   - Suporte inteligente para separar tags por vírgula (modelos como *Illustrious*) ou apenas por espaço (modelos como *Flux*).
   - **Vocabulário Global:** O sistema escaneia todos os seus arquivos de texto e mantém um banco de palavras globais, permitindo reaproveitar tags em novas imagens com apenas um clique esquerdo, ou deletá-las do dicionário com o clique direito.

2. **Espelhamento de Imagens em Lote (Flip):**
   - Inverta horizontalmente dezenas ou centenas de imagens simultaneamente e salve-as em uma pasta de saída automatizada, otimizando o preparo do seu dataset.

3. **Anotação de Textos em Lote:**
   - Crie novos arquivos `.txt` em lote para imagens recém-adicionadas, adicionando ou sobrescrevendo prompts comuns em todas elas de uma só vez.

## 🔮 O que vem por aí?

**Novas atualizações focadas no fluxo de trabalho e automação para geração de IAs e treinamento de LoRAs estão a caminho!** O projeto será expandido para ajudar ainda mais quem trabalha com edição e criação de modelos generativos.

## 🛠️ Instalação e Uso (Windows)
1. Clone este repositório ou faça o download.
2. Dê **duplo clique** no arquivo `run.bat`.
3. Na primeira execução, ele criará automaticamente o ambiente virtual, instalará as dependências (`PyQt6` e `Pillow`) e iniciará o aplicativo.
