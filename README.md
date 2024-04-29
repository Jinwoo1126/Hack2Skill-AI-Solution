## Hack2Skill hackathon AI Solution Team
We spent two weeks developing our AI solution for this [GenAI Hackathon APAC Edition](https://hack2skill.com/genai_hackathon_apac_edition?utm_medium=H2Ssocial&utm_source=H2Ssocial) to solve a real-world problem in retail and e-commerce. We've built cutting-edge applications to solve problems with tools like Vertex AI, Duet AI, GenAI App Builder, Model Garden, and more.

### Streamlit Demo
[Interior AI Solution](http://34.64.68.13:8502/)

### Set-up docker environment
1. create docker network 
    - Settings for network connectivity between containers
        ```
        $ sh create_network.sh
        ```
2. build docker images
    - build `streamlit` and `fastapi` docker images
        ```
        $ sh build_images.sh
        ```
3. run & delete container
    - run and delete containers by docker-compose
        ```
        $ docker compose up -d
        $ docker compose down
        ```

### Input Image
![Input Image](https://github.com/Jinwoo1126/Hack2Skill-AI-Solution/blob/main/input_image.png)

### Output Image
![Onput Image 1](https://github.com/Jinwoo1126/Hack2Skill-AI-Solution/blob/main/output_image_1.png)
![Onput Image 2](https://github.com/Jinwoo1126/Hack2Skill-AI-Solution/blob/main/output_image_2.png)
