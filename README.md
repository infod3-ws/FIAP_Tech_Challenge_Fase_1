# 📊 Tech Challenge — Fase 1 | NPS Preditivo

> Pós-Graduação em Data Science & Inteligência Artificial — FIAP PosTech
> Fase 1 — Fundamentos, Análise Exploratória e Introdução a Modelos Preditivos

-----

## 📌 Objetivo do Projeto

Este projeto foi desenvolvido como entrega do **Tech Challenge da Fase 1** do curso
de Pós-Graduação em Data Science e IA da FIAP PosTech.

O desafio simula um cenário real de e-commerce em crescimento acelerado, onde a
empresa enfrenta alta variabilidade no **Net Promoter Score (NPS)** — a métrica
que mede a lealdade e satisfação dos clientes.

O NPS é coletado apenas *após* o encerramento da jornada de compra, o que limita
a capacidade da empresa de agir de forma preventiva. O objetivo central do projeto é:

> **Identificar quais fatores operacionais realmente influenciam a satisfação do
> cliente e construir uma estratégia preditiva capaz de antecipar o risco de
> insatisfação antes da aplicação da pesquisa de NPS.**

### O projeto responde às seguintes perguntas

- Qual é o real estado do NPS da empresa — e o que os números escondem?
- Quais variáveis operacionais mais impactam a satisfação do cliente?
- Existe um “ponto de ruptura” na experiência — um limiar a partir do qual
  a insatisfação se torna irreversível?
- Como um modelo de Machine Learning pode identificar clientes em risco
  de se tornar detratores antes mesmo da pesquisa ser enviada?

-----

## 🗂️ Estrutura do Repositório

```
├── data/
│   └── desafio_nps_fase_1.csv        # Base de dados histórica de pedidos e NPS
│
├── notebooks/
│   ├── 01_EDA_NPS_Preditivo.ipynb    # Análise Exploratória de Dados (Requisito 3)
│   └── 02_Modelo_Preditivo.ipynb     # Modelo de Classificação com Random Forest (Requisito 4)
│
└── README.md
```

-----

## 🗃️ Descrição da Base de Dados

**Arquivo:** `data/desafio_nps_fase_1.csv`
**Registros:** 2.500 pedidos
**Variáveis:** 19 colunas
**Valores ausentes:** nenhum — base 100% completa

A base contém dados históricos de pedidos, entregas e interações com o atendimento
ao cliente, coletados após o encerramento da jornada de compra.

### Dicionário de Dados

|Variável                   |Tipo |Descrição                                    |
|---------------------------|-----|---------------------------------------------|
|`customer_id`              |int  |Identificador único do cliente               |
|`order_id`                 |int  |Identificador único do pedido                |
|`customer_age`             |int  |Idade do cliente (anos)                      |
|`customer_region`          |str  |Região geográfica do cliente                 |
|`customer_tenure_months`   |int  |Tempo de relacionamento com a empresa (meses)|
|`order_value`              |float|Valor total do pedido (R$)                   |
|`items_quantity`           |int  |Quantidade de itens no pedido                |
|`discount_value`           |float|Valor de desconto aplicado (R$)              |
|`payment_installments`     |int  |Número de parcelas do pagamento              |
|`delivery_time_days`       |int  |Tempo total de entrega (dias)                |
|`delivery_delay_days`      |int  |Dias de atraso em relação ao prazo prometido |
|`freight_value`            |float|Valor do frete (R$)                          |
|`delivery_attempts`        |int  |Número de tentativas de entrega              |
|`customer_service_contacts`|int  |Número de contatos com o atendimento (SAC)   |
|`resolution_time_days`     |int  |Tempo para resolução de problemas (dias)     |
|`complaints_count`         |int  |Número de reclamações registradas            |
|`repeat_purchase_30d`      |int  |Recompra em até 30 dias? (0 = não, 1 = sim)  |
|`csat_internal_score`      |float|Score interno de satisfação do cliente       |
|`nps_score`                |float|Nota NPS do cliente, de 0 a 10               |

### ⚠️ Nota sobre Data Leakage

As variáveis `repeat_purchase_30d` e `csat_internal_score` foram **excluídas**
do modelo preditivo por representarem consequências da satisfação, e não causas.
Incluí-las produziria resultados artificialmente inflados que não se sustentariam
em produção.

-----

## 🔬 Metodologia Utilizada

O projeto foi estruturado em duas grandes entregas, cada uma em um notebook separado.

### Notebook 1 — Análise Exploratória de Dados (EDA)

**Objetivo:** entender o estado atual do NPS e identificar os fatores operacionais
que mais impactam a satisfação do cliente, comunicando os achados em linguagem
acessível para gestores não técnicos.

**Etapas realizadas:**

1. **Diagnóstico inicial do NPS**
   Cálculo do NPS consolidado, distribuição por categoria (Detrator / Neutro / Promotor)
   e visualização da proporção de cada grupo.
1. **Ranking de correlações**
   Correlação de Pearson entre cada variável operacional e o `nps_score`,
   excluindo variáveis com risco de data leakage.
1. **Análise do atraso na entrega**
   NPS médio por dia exato de atraso, identificação do ponto de ruptura
   e distinção entre tempo total de entrega vs. dias de atraso.
1. **Análise de reclamações e SAC**
   NPS médio por número de reclamações e por número de contatos com o atendimento.
1. **Combinação de fatores**
   Heatmap da combinação atraso alto + SAC repetido para identificar
   o cenário mais destrutivo para o NPS.
1. **Fatores sem impacto**
   Análise de região geográfica e tempo de relacionamento,
   confirmando que não influenciam o NPS de forma relevante.
1. **Perfil do Detrator vs. Promotor**
   Comparativo das médias operacionais entre os dois grupos extremos.

-----

### Notebook 2 — Modelo Preditivo (Classificação)

**Objetivo:** construir um modelo de Machine Learning capaz de identificar
clientes com alto risco de se tornarem detratores antes da aplicação da pesquisa de NPS.

**Estratégia adotada:** Classificação Binária (Detrator vs. Não-Detrator)

A escolha por classificação binária em vez de regressão ou multiclasse foi motivada
pela necessidade de uma saída operacionalmente acionável: saber *quem está em risco*
é mais útil do que estimar uma nota exata.

**Etapas realizadas:**

1. **Definição da variável alvo**
   `target = 1` (Detrator) para `nps_score < 7`, e `target = 0` (Não-Detrator) para o restante.
1. **Seleção e preparação das features**
   14 variáveis operacionais selecionadas. Variável categórica `customer_region`
   transformada via Label Encoding.
1. **Separação treino / teste**
   80% para treino e 20% para teste, com estratificação pela variável alvo
   para manter a proporção de classes em ambos os conjuntos.
1. **Modelos avaliados**
   
   |Modelo             |Papel                         |
   |-------------------|------------------------------|
   |DummyClassifier    |Baseline — mínimo aceitável   |
   |Regressão Logística|Referência linear             |
   |**Random Forest**  |**Modelo principal escolhido**|
1. **Justificativa do Random Forest**
   Robusto a variáveis mistas, captura relações não-lineares (como o ponto de
   ruptura do atraso), oferece importância das features e lida bem com
   desbalanceamento via `class_weight='balanced'`.
1. **Métricas de avaliação**
   F1-Score (foco nos detratores), ROC-AUC, Matriz de Confusão
   e Validação Cruzada com 5 folds.
1. **Simulação prática**
   Demonstração de como o modelo classificaria pedidos individuais
   e quais ações operacionais seriam recomendadas em cada cenário.

-----

## ▶️ Como Reproduzir os Resultados

### Pré-requisitos

- Python 3.8 ou superior
- Conta no [Google Colab](https://colab.research.google.com/) (recomendado)
  ou ambiente local com Jupyter Notebook

### Bibliotecas necessárias

Todas as bibliotecas utilizadas estão disponíveis por padrão no Google Colab.
Se for rodar localmente, instale com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly
```

### Passo a passo

**Opção 1 — Google Colab (recomendado)**

1. Acesse [colab.research.google.com](https://colab.research.google.com/)
1. Clique em `Arquivo` → `Abrir notebook` → aba `GitHub`
1. Cole a URL deste repositório e selecione o notebook desejado
1. No menu lateral, clique no ícone de pasta e faça o upload do arquivo
   `data/desafio_nps_fase_1.csv`
1. Execute as células em sequência com `Shift + Enter` ou
   `Runtime` → `Run all`

**Opção 2 — Ambiente local**

```bash
# 1. Clone o repositório
git clone https://github.com/AnaRaquelCafe/POSTECH_AI_SCIENTIST.git
cd POSTECH_AI_SCIENTIST

# 2. Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn plotly

# 3. Abra o Jupyter
jupyter notebook

# 4. Navegue até a pasta /notebooks e abra o notebook desejado
```

### Ordem de execução recomendada

```
1. notebooks/01_EDA_NPS_Preditivo.ipynb
        ↓
2. notebooks/02_Modelo_Preditivo.ipynb
```

O notebook de EDA não depende do de modelo, mas a leitura em sequência
garante melhor compreensão do contexto antes da etapa preditiva.

> **Atenção:** certifique-se de que o arquivo `desafio_nps_fase_1.csv`
> está acessível no ambiente antes de executar qualquer célula de carregamento.
> 
> - **No Colab** (após upload manual): use o caminho `'desafio_nps_fase_1.csv'`
> - **Localmente** (com estrutura de pastas do repositório): ajuste para `'../data/desafio_nps_fase_1.csv'`

-----

## 📋 Principais Resultados

|Indicador              |Resultado                                                                                                                                         |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
|NPS Consolidado        |**−80 pontos**                                                                                                                                    |
|Detratores na base     |**84,4%** dos clientes                                                                                                                            |
|Fator com maior impacto|**Atraso na entrega** (correlação −0,60)                                                                                                          |
|Ponto de ruptura       |**3º dia de atraso** — queda acelerada e consistente                                                                                              |
|Insight contraintuitivo|Tempo total de entrega tem correlação de **0,0009** com o NPS — o que destrói a satisfação não é a demora, mas o descumprimento do prazo prometido|
|Modelo escolhido       |**Random Forest** com `class_weight='balanced'`                                                                                                   |
|Validação              |Consistência confirmada por **cross-validation com 5 folds**                                                                                      |

-----

## ⚠️ Limitações

- A base reflete apenas clientes que responderam à pesquisa de NPS,
  podendo apresentar viés de resposta
- O modelo foi treinado com dados históricos e deve ser retreinado
  periodicamente à medida que a operação evolui
- Correlação não implica causalidade: o modelo identifica padrões,
  não explica as causas profundas da insatisfação
- Com 84% de detratores na base, o desbalanceamento de classes foi tratado
  via `class_weight='balanced'`, mas deve ser monitorado em produção

-----

*Projeto desenvolvido para fins acadêmicos como parte do Tech Challenge
da Fase 1 da Pós-Graduação em Data Science & IA — FIAP PosTech.*
