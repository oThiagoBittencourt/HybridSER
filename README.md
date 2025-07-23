# HybridSER

Este repositório contém o material referente ao Trabalho de Conclusão de Curso (TCC) intitulado:

> **"Investigação de Modelos Híbridos entre Redes Convolucionais e Recorrentes para Reconhecimento de Emoções na Fala em Contexto Multilíngue"**

## 📘 Descrição

Este projeto propõe o desenvolvimento de um protótipo de _Speech Emotion Recognition_ (SER) com foco em **contextos multilíngues**, enfrentando desafios de variabilidade fonética e prosódica entre idiomas. A abordagem investiga **modelos híbridos baseados em redes neurais (CNN, RNN, LSTM, GRU)**, aliando **técnicas de aumento de dados** e **extração de atributos acústicos** (MFCC, Chromagram, ZCR, RMS).

## 🎯 Objetivos

### Geral

- Desenvolver um sistema inteligente para reconhecimento de emoções a partir da fala em contextos multilíngues.

### Específicos

- Investigar técnicas de extração de atributos e data augmentation.
- Implementar e comparar modelos híbridos CNN+(RNN/LSTM/GRU).
- Avaliar a performance com métricas como F1-Score, Precisão, Revocação e validação cruzada.

![Figura 21 – Arquitetura do modelo de SER](https://github.com/user-attachments/assets/1f428711-d4ef-4819-813c-ccf1afec4369)

## 🧰 Tecnologias e Ferramentas

- Python 3.11
- TensorFlow / Keras
- Librosa
- Scikit-learn

## 🗂 Bases de Dados

- **VERBO** (português)
- **CaFE** (francês canadense)
- **RAVDESS** (inglês norte-americano)

## 📁 Estrutura do Projeto

```text
HybridSER/
├── src/                → Código-fonte principal
│   ├── data/           → Carregamento de datasets
│   ├── features/       → Extração de atributos de áudio
│   ├── models/         → Modelos híbridos (CNN + RNN/LSTM/GRU)
│   └── utils/          → Funções auxiliares
├── tests/              → Testes automatizados (pytest)
├── .github/workflows/  → Pipeline CI com flake8 + pytest
├── .vscode/            → Configurações automáticas do VSCode
├── requirements.txt    → Dependências do projeto
├── .flake8             → Regras do linter
├── .gitignore          → Arquivos e pastas ignorados
├── README.md           → Visão geral do projeto
├── SETUP.md            → Guia de instalação
```

## 👨‍🎓 Autores

- Gabriel Martins Delfes
- Thiago Martins Escaliante
- Gael Huk Kukla
- Felipe Franco Pinheiro
- Yann Lucas Saito da Luz
- Thiago Bittencourt Santana

**Orientador:** Leandro Fabian Almeida Escobar

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

> "Em toda a filosofia humana, em toda a ciência, há sempre uma ideia fundamental - variável segundo os sistemas e as ciências - que nos esquecemos de provar."
> — _Fernando Pessoa_
