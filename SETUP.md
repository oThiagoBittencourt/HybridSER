## 📄 Setup HybridSER

Projeto de Reconhecimento de Emoções na Fala (SER) com modelos híbridos (CNN + RNN/LSTM/GRU), aplicado em contextos multilíngues.

---

### ✅ Requisitos

* Python 3.11.9
* Git
* VSCode (opcional)

---

### ⚙️ Instalação do Ambiente

```bash
# Clone o repositório
git clone https://github.com/oThiagoBittencourt/HybridSER.git
cd HybridSER

# Crie o ambiente virtual
python -m venv .venv

# Ative o ambiente virtual:
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# ou Windows (CMD)
.venv\Scripts\activate.bat

# ou Linux/macOS
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

---

### 🧪 Rodar os testes com Pytest

```bash
pytest
```

---

### 🧹 Verificar estilo com Flake8

```bash
flake8 src/
```

---

### 💡 Configurar VSCode (opcional)

Se estiver usando VSCode:

1. Pressione `Ctrl + Shift + P`
2. Escolha `Python: Select Interpreter`
3. Selecione `.venv\Scripts\python.exe`

O VSCode já vem pré-configurado com o arquivo `.vscode/settings.json` para lint e testes.

---

### 📁 Estrutura do Projeto

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
