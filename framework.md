```mermaid
graph TD;
    subgraph Insert prompt
    A[Prompt];
    style A fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000000;
    end

    subgraph Prompt Evaluation
        A -.-> |**Jump Start** 
        Get predictions on dev set| B[Predictor];
        B --> |Evaluate the dev set predictions| C[Evaluator];
        style B fill:#ffcc00,stroke:#333,stroke-width:2px,color:#000000;
        style C fill:#ff9999,stroke:#333,stroke-width:2px,color:#000000;
    end

    subgraph Training Step
        A --> |Get predictions on a batch of training set| D[Predictor];
        D --> |Evaluate batch predictions| E[Evaluator];
        style D fill:#ffcc00,stroke:#333,stroke-width:2px,color:#000000;
        style E fill:#ff9999,stroke:#333,stroke-width:2px,color:#000000;
    end

    subgraph Optimization Step
        D --> |Get Feedback on Predictions| F[Feedback Generator];
        E --> |Get Feedback on Predictions| F;
        F --> |Use the feedback to get a new prompt| Q[Prompt Generator];
        Q --> B;
        style F fill:#ccffcc,stroke:#333,stroke-width:2px,color:#000000;
        style Q fill:#ffccff,stroke:#333,stroke-width:2px,color:#000000;
    end

    subgraph Search Algorithm
        C --> |Record the prompt and its value| R[Record];
        R --> |Look up the records and suggest the next prompt| N[Return Next Prompt];
        N --> A;
        style R fill:#ccccff,stroke:#333,stroke-width:2px,color:#000000;
        style N fill:#ffcc99,stroke:#333,stroke-width:2px,color:#000000;
    end
```

<!-- Color Palette:
- Prompt: #f9f9f9
- Predictor: #ffcc00
- Evaluator: #ff9999
- Feedback Generator: #ccffcc
- Prompt Generator: #ffccff
- Record: #ccccff
- Return Next Prompt: #ffcc99
- Insert prompt subgraph: #ffffff
- Prompt Evaluation subgraph: #e6f7ff
- Training Step subgraph: #e6ffe6
- Optimization Step subgraph: #fff0e6
- Search Algorithm subgraph: #f2e6ff
-->
