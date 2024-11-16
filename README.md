# **Projeto: Leitor de QR Code com Identificação de Deslocamento**

### **Descrição**

Este projeto utiliza Python e OpenCV para criar um software capaz de:

1. Ler QR codes a partir de uma imagem ou vídeo.
2. Identificar o deslocamento do QR code em termos de pixels.
3. Converter o deslocamento de pixels para metros, usando uma referência calibrada.

### **Requisitos**

- Python 3.7 ou superior
- Bibliotecas necessárias:
  - `opencv-python`
  - `opencv-contrib-python`
  - `numpy`

### **Instalação**

1. Clone o repositório:

   ```bash
   git clone https://github.com/itsmewall/comp_vision
   ```
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

### **Uso**

1. Execute o script principal:

   ```bash
   python main.py
   ```
2. Insira uma imagem ou vídeo contendo o QR code.
3. O software detectará o QR code, calculará o deslocamento em pixels e exibirá a conversão para metros.

### **Estrutura do Projeto**

- `main.py`: Script principal que implementa a funcionalidade.
- `calibration_data/`: Contém arquivos para calibração de câmera (se necessário).
- `README.md`: Instruções do projeto.

### **Notas**

- Certifique-se de calibrar sua câmera com uma referência conhecida (como o tamanho físico do QR code) para garantir precisão na conversão de pixels para metros.
