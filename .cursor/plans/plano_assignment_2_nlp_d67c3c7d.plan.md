---
name: Plano assignment 2 NLP
overview: Construir um roteiro para desenvolver o segundo assignment usando Hugging Face transformers na mesma tarefa de classificação do primeiro assignment, reaproveitando dados, métricas e baselines já existentes no repositório.
todos:
  - id: freeze-baseline
    content: Consolidar a baseline oficial do Assignment 1 a partir dos CSV em `results`.
    status: pending
  - id: prepare-transformer-data
    content: Definir input textual e preparar os mesmos splits do Assignment 1 para treino com transformers.
    status: pending
  - id: select-hf-models
    content: Escolher 2 a 3 modelos Hugging Face adequados ao cenário multilingue.
    status: pending
  - id: build-first-pipeline
    content: Montar o primeiro pipeline de fine-tuning e avaliação com Hugging Face.
    status: pending
  - id: run-core-experiments
    content: Executar uma grelha pequena de experiências e registar métricas comparáveis.
    status: pending
  - id: error-analysis
    content: Analisar erros do melhor transformer e contrastar com os modelos clássicos.
    status: pending
  - id: bonus-if-time
    content: Explorar apenas um bónus opcional, como LoRA/PEFT ou domain adaptation.
    status: pending
  - id: report-packaging
    content: Organizar resultados, figuras e narrativa final para relatório/apresentação.
    status: pending
isProject: false
---

# Planificação Step by Step

## Objetivo

Fazer o segundo assignment sobre a mesma tarefa de classificação (`is_safe`) do dataset Salamandra Guard, mas agora com fine-tuning de modelos Hugging Face, comparando-os de forma justa com os melhores modelos clássicos já obtidos em [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/model_results.csv](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/model_results.csv) e [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_model_results.csv](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_model_results.csv).

## 1. Rever e congelar a baseline do Assignment 1

- Confirmar exatamente qual foi a melhor baseline clássica e com que configuração:
  - dataset variante: `resp_no_punct`, `comb_no_punct`, etc.
  - modelo: Logistic Regression vs Linear SVM
  - métricas finais: accuracy, precision, recall, macro-F1
- Usar como referência os ficheiros:
  - [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/model_results.csv](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/model_results.csv)
  - [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_model_results.csv](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_model_results.csv)
  - [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_best_params.csv](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/results/tuned_best_params.csv)
- Resultado esperado desta etapa: uma tabela curta com a baseline oficial a bater.

## 2. Definir a estratégia de dados para transformers

- Partir do mesmo dataset/tarefa usado no primeiro assignment: Salamandra Guard com target `is_safe`.
- Decidir qual input vais usar no transformer:
  - só `response`
  - `prompt + response`
- Começar com a opção que deu melhor desempenho no assignment 1; só depois testar a alternativa se houver tempo.
- Evitar pré-processamento agressivo de [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/src/preprocessing.py](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/src/preprocessing.py):
  - não remover stopwords
  - não lematizar
  - não destruir pontuação por defeito
- No máximo, aplicar limpeza mínima: espaços duplicados, nulos, normalização básica.
- Reutilizar os splits do trabalho anterior para garantir comparação justa. Se os CSV processados estiverem incompletos, reconstruir a partir de [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/data/raw/](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/data/raw/).

## 3. Escolher 2 a 3 modelos Hugging Face candidatos

- Como os dados são multilingues (sobretudo catalão e espanhol), começar por modelos multilingues.
- Ordem recomendada:
  1. `xlm-roberta-base` como candidato principal
  2. `bert-base-multilingual-cased` como baseline transformer forte e leve
  3. opcional: um modelo mais pequeno/distilled para comparar custo vs desempenho
- Justificação a escrever no relatório:
  - cobertura multilingue
  - adequação a classificação de texto
  - equilíbrio entre qualidade e custo computacional
- Se quiseres um bónus mais ambicioso, reservar depois uma experiência com PEFT/LoRA ou domain adaptation, mas não como primeiro passo.

## 4. Montar o pipeline mínimo com Hugging Face

- Criar um notebook ou script novo dedicado ao assignment 2, separado dos notebooks clássicos. Exemplo natural:
  - [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/notebooks/](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/notebooks/) com um novo notebook `08.Transformers.ipynb`
- O pipeline deve ter estas etapas:
  - carregar dataset
  - preparar texto final
  - tokenizar com `AutoTokenizer`
  - mapear labels para `0/1`
  - carregar `AutoModelForSequenceClassification`
  - treinar com `Trainer` ou `TrainingArguments`
  - avaliar no mesmo teste do assignment 1
- Guardar resultados numa estrutura comparável à dos CSV atuais em `results`.

## 5. Definir uma grelha de experiências pequena e realista

- Não tentes explorar demasiadas combinações logo no início.
- Ordem recomendada dos testes:
  1. `bert-base-multilingual-cased` com `response`
  2. `xlm-roberta-base` com `response`
  3. melhor modelo acima com `prompt + response`
  4. opcional: ajuste fino de hiperparâmetros do melhor modelo
- Hiperparâmetros a variar de forma controlada:
  - learning rate
  - batch size
  - número de epochs
  - max sequence length
  - weight decay
- Fixar seeds e documentar tudo para reprodutibilidade.

## 6. Avaliar com as mesmas métricas do Assignment 1

- Medir pelo menos:
  - accuracy
  - precision
  - recall
  - F1
  - macro-F1
- Se possível, acrescentar:
  - confusion matrix
  - classificação por língua (`language`) para perceber se o modelo falha mais em catalão ou espanhol
- Resultado esperado: uma tabela comparativa única com clássicos vs transformers.

## 7. Fazer análise de erros orientada

- Repetir a lógica do notebook [C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/notebooks/07.ErrorAnalysis.ipynb](C:/Users/Lenovo/Desktop/feup/Master/PLN/NLP-project1/notebooks/07.ErrorAnalysis.ipynb), mas agora para o melhor transformer.
- Analisar manualmente exemplos de:
  - falsos positivos
  - falsos negativos
  - erros por língua
  - erros em prompts ambíguos ou respostas longas
- Objetivo: perceber se o transformer melhora contexto/nuance face ao TF-IDF ou se continua a falhar nos mesmos padrões.

## 8. Só depois explorar bónus

- Se o modelo base já estiver funcional e avaliado, escolher apenas um extra:
  - LoRA / PEFT para treino mais leve
  - domain adaptation adicional no corpus do domínio
  - prompting com LLM como comparação exploratória
- Escolhe um só bónus para não dispersar o projeto.

## 9. Organizar os artefactos finais

- Produzir no mínimo:
  - notebook/script de treino transformer
  - ficheiro de resultados comparativos em `results`
  - figuras finais para relatório
  - secção metodológica clara a justificar escolha do modelo
- Idealmente manter a mesma lógica organizacional do repositório atual:
  - `data` para dados
  - `notebooks` para exploração/experiências
  - `src` para funções reutilizáveis
  - `results` para métricas finais

## 10. Estrutura recomendada para o relatório

- Introdução curta: tarefa, dataset, objetivo do assignment 2
- Motivação para escolher transformers em vez de features manuais
- Escolha do modelo pré-treinado e justificação linguística
- Preparação dos dados e estratégia experimental
- Resultados quantitativos comparados com o assignment 1
- Análise de erros
- Limitações e trabalho futuro

## Ordem prática de execução

1. Confirmar baseline clássica final.
2. Preparar dataset sem pré-processamento destrutivo.
3. Testar `bert-base-multilingual-cased`.
4. Testar `xlm-roberta-base`.
5. Comparar métricas com os CSV do assignment 1.
6. Melhorar só o melhor modelo.
7. Fazer análise de erros.
8. Se houver tempo, adicionar um bónus.
9. Fechar relatório e apresentação.

## Critério de sucesso

O projeto fica bem encaminhado se no fim tiveres:

- um pipeline reprodutível com Hugging Face
- pelo menos 2 modelos transformer comparados
- comparação direta com a melhor baseline clássica
- análise de erros convincente
- justificação clara das escolhas metodológicas
