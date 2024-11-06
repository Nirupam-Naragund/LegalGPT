# LegalGPT

**LegalGPT** is a specialized chatbot designed to assist users with legal queries specifically related to Indian law. It can provide legal information and answer questions focused on the Indian legal system. For non-legal or non-Indian-specific queries, it will respond with a message indicating that it is unable to assist. This ensures LegalGPT remains targeted and relevant to the legal domain in India.

LegalGPT is built using **Mixtral**, an open-source Large Language Model (LLM) that allows it to generate accurate responses tailored to legal requirements in India.

## Features

- Provides answers to Indian law-related questions.
- Restricts responses to the legal domain within India.
- Responds with a polite disclaimer if asked questions outside its scope.

## Tech Stack

- **LLM**: Mixtral (Open-source)
- **Frontend**: Streamlit (for interactive interface)

## Getting Started

To set up and run LegalGPT locally, follow the steps below.

### Prerequisites

- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/your-username/LegalGPT.git
    cd LegalGPT
    ```

2. **Create a Virtual Environment**:

    Create a virtual environment to keep dependencies organized.

    ```bash
    python3 -m venv legalgpt-env
    ```

3. **Activate the Virtual Environment**:

    - **Windows**:

      ```bash
      legalgpt-env\Scripts\activate
      ```

    - **Mac/Linux**:

      ```bash
      source legalgpt-env/bin/activate
      ```

4. **Install Required Packages**:

    Use `requirements.txt` to install all necessary dependencies.

    ```bash
    pip install -r requirements.txt
    ```

5. **Set Up Environment Variables**:

    Create a `.env` file in the root directory, following the format provided in `env.example`. Ensure you add any necessary API keys and configurations.

6. **Run the Application**:

    Start LegalGPT by running the following command:

    ```bash
    streamlit run main.py
    ```

## Usage

Once the application is running, open your browser and navigate to the URL provided by Streamlit to start interacting with LegalGPT.

## Contributing

Contributions to LegalGPT are welcome! Feel free to submit issues or pull requests to improve the chatbot or expand its functionality.

## License

This project is licensed under the MIT License.
