
const definitionsFr = [
    // les définitions en français ici
    {
        title: "Apprentissage supervisé",
        description: "L’apprentissage supervisé est un procédé d’apprentissage automatique dans lequel l’algorithme s’entraîne à une tâche déterminée en utilisant un jeu de données assorties chacune d’une annotation indiquant le résultat attendu. Dans l'apprentissage supervisé, nous utilisons beaucoup les algorithmes de classification, qui cherchent à prédire une classe/catégorie.",
        example: "Exemple :",
        image: ""
    },
    {
        title: "Apprentissage non supervisé",
        description: "L’apprentissage non supervisé est un procédé d’apprentissage automatique dans lequel l’algorithme utilise un jeu de données brutes et obtient un résultat en se fondant sur la détection de similarités entre certaines de ces données. Les algorithmes de ce type d'apprentissage peuvent être utilisés pour trois types en problèmes.",
        example: "Exemple :",
        image: ""
    },
    {
        title: "Chatbot",
        description: "Un programme informatique qui simule une conversation humaine pour fournir une assistance ou des informations aux utilisateurs. On pourrait penser par exemple à Siri d’Apple !",
        example: "",
        image: ""
    },
    {
        title: "Embedding",
        description: "Représentation vectorielle d'un mot, d'une phrase ou d'un document dans un espace multidimensionnel, utilisée pour enseigner la sémantique à un modèle de langage. Entre autre, l’embedding, c’est le fait de représenter un mot par des chiffres et de le représenter dans un espace vectoriel, plus ou moins distant des mots qui lui ressemble. Par exemple, on pourrait s’attendre à ce que les mots « satisfait » et « heureux » soient représentés par des vecteurs relativement peu distants dans l’espace vectoriel où sont définis ces vecteurs.",
        example: "",
        image: ""
    },
    {
        title: "Entraînement",
        description: "La phase d'apprentissage durant laquelle un modèle de langage apprend à partir de données pour comprendre et générer du langage.",
        example: "",
        image: ""
    },
    {
        title: "Fine-tuning",
        description: "Le processus d'ajustement d'un modèle de langage pré-entraîné pour qu'il soit mieux adapté à une tâche spécifique ou à un domaine particulier en continuant son entraînement sur un ensemble de données ciblé. C’est comme si nous entrainions un chien pour un service spécial, autre que simplement assis, couché, etc.",
        example: "",
        image: ""
    },
    {
        title: "Génération de texte",
        description: "La capacité d'un modèle de langage à créer du texte de manière autonome, souvent utilisée pour écrire des articles, composer des emails ou générer des dialogues.",
        example: "",
        image: ""
    },
    {
        title: "GPT (Generative Pre-trained Transformer)",
        description: "Type de LLM conçu pour générer des textes cohérents en prédisant la suite logique dans une phrase donnée.",
        example: "",
        image: ""
    },
    {
        title: "GPU (Graphics Processing Units)",
        description: "L'un des types de technologie informatique les plus importants, tant pour l'informatique personnelle que pour l'informatique d'entreprise. Conçu pour le traitement parallèle, le GPU est utilisé dans un large éventail d'applications, notamment le graphisme et le rendu vidéo. Bien qu'ils soient surtout connus pour leurs capacités dans le domaine des jeux, les GPU sont de plus en plus utilisés dans la production créative et l'intelligence artificielle (IA). Par exemple, pour créer GPT-3 il a fallu 1024 GPU qui travaillent durant 34 jours, tout cela coûtant plusieurs millions d’euros.",
        example: "",
        image: ""
    },
    {
        title: "Hallucinations",
        description: "Phénomène où un modèle de langage génère des réponses erronées ou hors sujet avec une apparence de certitude.",
        example: "",
        image: ""
    },
    {
        title: "HuggingFace",
        description: "Hugging Face est une entreprise fournissant des bibliothèques open source contenant des modèles d’IA pré-formés. Nous pouvons donc des modèles innovant entraîné par les passionés et la communauté, ce qui en fait une mine d’or pour les chercheurs. Nous pouvons y retrouver un classement des meilleurs modèles, actualisé tous les jours.",
        example: "",
        image: ""
    },
    {
        title: "Hyperparamètres",
        description: "Les configurations qui déterminent la structure du modèle et la manière dont l'entraînement est conduit, comme le taux d'apprentissage ou la taille du lot (batch size). Ces paramètres ne sont généralement pas connus du grand public, seuls les développeurs les modifient.",
        example: "",
        image: ""
    },
    {
        title: "IA générative",
        description: "Catégorie de l'intelligence artificielle qui crée de nouveaux contenus (texte, images, code, etc.) en s'appuyant sur des modèles complexes et des données existantes, comme Chat GPT par exemple.",
        example: "",
        image: ""
    },
    {
        title: "Large Language Model (LLM)",
        description: "Un programme informatique avancé conçu pour comprendre, interpréter et générer du langage humain en se basant sur de vastes quantités de données textuelles.",
        example: "",
        image: ""
    },
    {
        title: "LLAMA",
        description: "LLaMA, pour Large Language Model Meta AI, est, comme son nom l’indique, un modèle linguistique. Il est OpenSource !",
        example: "",
        image: ""
    },
    {
        title: "Machine Learning",
        description: "Multimodal : Se réfère à des modèles capables de comprendre et de générer différents types de contenu, comme le texte, les images et le code.",
        example: "",
        image: ""
    },
    {
        title: "NER (Named Entity Recognition)",
        description: "Une technique d'apprentissage automatique utilisée pour identifier et classer les entités nommées (personnes, lieux, organisations, etc.) dans un texte.",
        example: "",
        image: ""
    },
    {
        title: "Paramètres du modèle (Model Parameters)",
        description: "Les éléments configurables (ou NON) d'un modèle de langage qui sont ajustés pendant l'entraînement pour améliorer sa performance.",
        example: "",
        image: ""
    },
    {
        title: "Parameter Efficient Fine-Tuning (PEFT)",
        description: "Ensemble de techniques permettant de modifier un petit nombre de paramètres d'un modèle préentraîné pour des tâches spécifiques sans réentraîner l'ensemble du modèle.",
        example: "",
        image: ""
    },
    {
        title: "Préentraînement",
        description: "Phase initiale d'entraînement d'un modèle de deep learning sur un large corpus de textes non étiquetés pour apprendre la structure du langage.",
        example: "",
        image: ""
    },
    {
        title: "Prompt",
        description: "Instruction ou question en langage naturel donnée à un modèle de langage pour générer une réponse ou un contenu spécifique.",
        example: "",
        image: ""
    },
    {
        title: "Prompt Engineering",
        description: "Pratique consistant à formuler des instructions précises pour guider un modèle de langage dans la génération de réponses souhaitées.",
        example: "",
        image: ""
    },
    {
        title: "Quantité de données (Dataset)",
        description: "Un ensemble de données textuelles utilisé pour entraîner ou fine-tuner un modèle de langage.",
        example: "",
        image: ""
    },
    {
        title: "Reinforcement Learning with Human Feedback (RLHF)",
        description: "Méthode d'apprentissage par renforcement où les modèles sont ajustés en fonction des retours et préférences humaines.",
        example: "",
        image: ""
    },
    {
        title: "Réseau de neurones (Neural Network)",
        description: "Une structure informatique qui imite le fonctionnement du cerveau humain pour traiter des informations et apprendre à partir de données.",
        example: "",
        image: ""
    },
    {
        title: "Retrieval Augmented Generation (RAG)",
        description: "Approche combinant un modèle de langage avec une recherche dans une base de données pour enrichir les réponses générées avec des informations pertinentes.",
        example: "",
        image: ""
    },
    {
        title: "Transformer",
        description: "Architecture de réseau de neurones spécialisée dans le traitement du langage naturel, utilisant un mécanisme d'autoattention pour gérer les relations entre les mots dans un texte.",
        example: "",
        image: ""
    },
    {
        title: "Token",
        description: "Plus petite unité de texte traitée par un modèle de langage, pouvant être un mot, une partie de mot ou un caractère de ponctuation.",
        example: "",
        image: ""
    }

];

const definitionsEn = [
    // les définitions en anglais ici
    
    {
        "title": "Supervised Learning",
        "description": "Supervised learning is a machine learning process in which the algorithm is trained on a specific task using a dataset where each entry is paired with an annotation indicating the expected result. In supervised learning, we often use classification algorithms, which aim to predict a class/category.",
        "example": "Example:",
        "image": ""
    },
    {
        "title": "Unsupervised Learning",
        "description": "Unsupervised learning is a machine learning process in which the algorithm uses raw data and obtains a result based on the detection of similarities between some of these data. Algorithms of this type of learning can be used for three types of problems.",
        "example": "Example:",
        "image": ""
    },
    {
        "title": "Chatbot",
        "description": "A computer program that simulates human conversation to provide assistance or information to users. One might think, for example, of Apple's Siri!",
        "example": "",
        "image": ""
    },
    {
        "title": "Embedding",
        "description": "Vector representation of a word, sentence, or document in a multidimensional space, used to teach semantics to a language model. Among other things, embedding is the act of representing a word by numbers and representing it in a vector space, more or less distant from words that resemble it. For example, one might expect the words \"satisfied\" and \"happy\" to be represented by vectors that are relatively close together in the vector space where these vectors are defined.",
        "example": "",
        "image": ""
    },
    {
        "title": "Training",
        "description": "The learning phase during which a language model learns from data to understand and generate language.",
        "example": "",
        "image": ""
    },
    {
        "title": "Fine-tuning",
        "description": "The process of adjusting a pre-trained language model to be better suited to a specific task or domain by continuing its training on a targeted dataset. It's like we're training a dog for a special service, other than just sitting, lying down, etc.",
        "example": "",
        "image": ""
    },
    {
        "title": "Text Generation",
        "description": "The ability of a language model to create text autonomously, often used to write articles, compose emails, or generate dialogues.",
        "example": "",
        "image": ""
    },
    {
        "title": "GPT (Generative Pre-trained Transformer)",
        "description": "Type of LLM designed to generate coherent text by predicting the logical sequence in a given sentence.",
        "example": "",
        "image": ""
    },
    {
        "title": "GPU (Graphics Processing Units)",
        "description": "One of the most important types of computing technology, both for personal and enterprise computing. Designed for parallel processing, the GPU is used in a wide range of applications, including graphics and video rendering. While best known for their capabilities in gaming, GPUs are increasingly being used in creative production and artificial intelligence (AI). For example, to create GPT-3 it took 1024 GPUs working for 34 days, all costing several million euros.",
        "example": "",
        "image": ""
    },
    {
        "title": "Hallucinations",
        "description": "Phenomenon where a language model generates incorrect or off-topic responses with an appearance of certainty.",
        "example": "",
        "image": ""
    },
    {
        "title": "HuggingFace",
        "description": "Hugging Face is a company providing open source libraries containing pre-trained AI models. So we can have innovative models trained by enthusiasts and the community, making it a gold mine for researchers. We can find a ranking of the best models, updated daily.",
        "example": "",
        "image": ""
    },
    {
        "title": "Hyperparameters",
        "description": "Configurations that determine the model structure and how training is conducted, such as learning rate or batch size. These parameters are generally not known to the general public, only developers modify them.",
        "example": "",
        "image": ""
    },
    {
        "title": "Generative AI",
        "description": "Category of artificial intelligence that creates new content (text, images, code, etc.) based on complex models and existing data, such as Chat GPT for example.",
        "example": "",
        "image": ""
    },
    {
        "title": "Large Language Model (LLM)",
        "description": "An advanced computer program designed to understand, interpret, and generate human language based on vast amounts of textual data.",
        "example": "Examples of LLMs",
        "image": ""
    },
    {
        "title": "LLAMA",
        "description": "LLaMA, which stands for Large Language Model Meta AI, is, as its name suggests, a language model. It is OpenSource!",
        "example": "",
        "image": ""
    },
    {
        "title": "Machine Learning",
        "description": "Multimodal: Refers to models capable of understanding and generating different types of content, such as text, images, and code.",
        "example": "",
        "image": ""
    },
    {
        "title": "NER (Named Entity Recognition)",
        "description": "A machine learning technique used to identify and classify named entities (people, places, organizations, etc.) in text.",
        "example": "",
        "image": ""
    },
    {
        "title": "Model Parameters",
        "description": "The configurable (or NOT) elements of a language model that are adjusted during training to improve its performance.",
        "example": "",
        "image": ""
    },
    {
        "title": "Parameter Efficient Fine-Tuning (PEFT)",
        "description": "A set of techniques for modifying a small number of parameters of a pretrained model for specific tasks without retraining the entire model.",
        "example": "",
        "image": ""
    },
    {
        "title": "Pretraining",
        "description": "Initial training phase of a deep learning model on a large corpus of unlabeled text to learn the structure of language.",
        "example": "",
        "image": ""
    },
    {
        "title": "Prompt",
        "description": "Natural language instruction or question given to a language model to generate a specific response or content.",
        "example": "",
        "image": ""
    },
    {
        "title": "Prompt Engineering",
        "description": "The practice of formulating precise instructions to guide a language model in generating desired responses.",
        "example": "",
        "image": ""
    },
    {
        "title": "Dataset",
        "description": "A set of textual data used to train or fine-tune a language model.",
        "example": "",
        "image": ""
    },
    {
        "title": "Reinforcement Learning with Human Feedback (RLHF)",
        "description": "Reinforcement learning method where models are adjusted based on human feedback and preferences.",
        "example": "",
        "image": ""
    },
    {
        "title": "Neural Network",
        "description": "A computer structure that mimics the functioning of the human brain to process information and learn from data.",
        "example": "Basis of everything",
        "image": ""
    },
    {
        "title": "Retrieval Augmented Generation (RAG)",
        "description": "An approach that combines a language model with database retrieval to enrich generated responses with relevant information.",
        "example": "Image",
        "image": ""
    },
    {
        "title": "Transformer",
        "description": "A neural network architecture specialized in natural language processing, using a self-attention mechanism to manage relationships between words in text.",
        "example": "",
        "image": ""
    },
    {
        "title": "Token",
        "description": "The smallest unit of text processed by a language model, which can be a word, part of a word, or punctuation mark.",
        "example": "",
        "image": ""
    }


];

const lexiqueSection = document.getElementById('lexique');
const btnFr = document.getElementById('btn-fr');
const btnEn = document.getElementById('btn-en');

function displayDefinitions(definitions) {
    lexiqueSection.innerHTML = '';
    definitions.sort((a, b) => a.title.localeCompare(b.title));
    definitions.forEach(def => {
        const defDiv = document.createElement('div');
        defDiv.classList.add('definition');
        
        const title = document.createElement('h2');
        title.textContent = def.title;
        defDiv.appendChild(title);
        
        const description = document.createElement('p');
        description.textContent = def.description;
        defDiv.appendChild(description);
        
        if (def.example) {
            const example = document.createElement('p');
            example.textContent = def.example;
            defDiv.appendChild(example);
        }
        
        if (def.image) {
            const image = document.createElement('img');
            image.src = def.image;
            defDiv.appendChild(image);
        }
        
        lexiqueSection.appendChild(defDiv);
    });
}

btnFr.addEventListener('click', () => displayDefinitions(definitionsFr));
btnEn.addEventListener('click', () => displayDefinitions(definitionsEn));

// Afficher les définitions en français par défaut
displayDefinitions(definitionsFr);
