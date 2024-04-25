## Hack2Skill hackathon AI Solution Team


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
