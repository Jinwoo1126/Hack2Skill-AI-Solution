import pandas as pd
from typing import Optional, Dict, List, Union

from google.oauth2 import service_account
from google.cloud import storage, aiplatform_v1, vision, aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.generative_models import Image as G_Image
from vertexai.preview.vision_models import Image, ImageGenerationModel

from PIL import Image, ImageDraw

from util import *
from variable import args
from indexing import EmbeddingResponse, EmbeddingPredictionClient


class Client:
    def __init__(self, args, genai_model:str="gemini-1.0-pro-vision") -> None:
        client_options = {"api_endpoint": args.API_ENDPOINT}

        self.args = args
        self._credentials = service_account.Credentials.from_service_account_file(
            args.KEY_PATH, scopes=args.SCOPES)

        # multimodal embedding client
        self.embedding = EmbeddingPredictionClient(project=args.PROJECT_ID)
        
        # vision client
        self.vision_ai = vision.ImageAnnotatorClient(credentials=self._credentials)
        
        # gcs client
        self.storage = storage.Client(credentials=self._credentials)
        self.bucket = self.storage.bucket(args.GCS_BUCKET)
        
        # vertex ai client to do similarity matching
        self.index = vector_search_client = aiplatform_v1.MatchServiceClient(
            client_options=client_options, credentials=self._credentials
        )

        # alloyDB connection
        self.engine, self.connector = None, None
        
        # generative model (gemini)
        vertexai.init(project=args.PROJECT_ID, credentials=self._credentials)
        self.gemini = GenerativeModel(genai_model)

    def connect_database(self) -> None:
        self.engine, self.connector = create_sqlalchemy_engine(self.args, self._credentials)
        return

    def detect_objects(self, image:Image):
        # content = image_file.read()
        content = image2bytes(image)
        vis_image = vision.Image(content=content)
        objects = self.vision_ai.object_localization(image=vis_image).localized_object_annotations
        return objects
    
    def draw_bbox(self, image:Image) -> (np.array, list):
        objects = self.detect_objects(image)
        
        w, h = image.width, image.height
        draw = ImageDraw.Draw(image)
        obj_info = {}
        for object_ in objects:
            box_text = f"{object_.name}: {object_.score:.2f}"
            box_coords = object_.bounding_poly.normalized_vertices
            draw.text(
                (box_coords[0].x * w, box_coords[0].y*h),
                box_text,
                fill=(255,255,0,255)
            )
            polygon = [[coord.x * w, coord.y * h] for coord in box_coords]
            draw.polygon(sum(polygon, []), None, 'red')
            obj_info[object_.name] = tuple(polygon[0] + polygon[2])
        return np.array(image), obj_info

    def get_recommended_items(self, 
                              text:Optional[str]=None, 
                              image_bytes:bytes=None,
                              threshold:float=0.7,
                              max_item:int=5,
                              search_type:str='Image'
                             ) -> (dict, pd.DataFrame, dict):
        
        if not text and not image_bytes:
            raise ValueError('At least one of text or image_bytes must be specified.')

        generated_text = {}
        if search_type.lower() == 'image':
            query_embeddings = self.embedding.get_embedding(image_bytes=image_bytes)
            query_embedding = '['+','.join(map(str, query_embeddings.image_embedding))+']'
        elif search_type.lower() == 'text':
            generated_text = self.get_text_query(image_bytes=image_bytes, predefined_prompt=text)
            search_term = 'a picture of ' + generated_text['keyword']
            query_embeddings = self.embedding.get_embedding(text=search_term)
            query_embedding = '['+','.join(map(str, query_embeddings.text_embedding))+']'
            
        values = {'value': query_embedding, 'max_item': max_item}
        self.connect_database()
        sql = get_sql()
        with self.engine.connect() as conn:
            result = conn.execute(sql, values)
        
        response = get_result_table(args, result)
        response = response.loc[response.SIMILARITY >= threshold].reset_index(drop=True)
        
        results = {}
        for i, row in response.iterrows():
            item_id = row.ITEM_ID
            path = f"{args.IMG_DIR}/{item_id}.jpg"
            if row.SIMILARITY >= threshold:
                # results.append(self.bucket.blob(path).download_as_bytes())
                results[item_id] = self.bucket.blob(path).download_as_bytes()

        response = response.drop('SIMILARITY', axis=1)
        return results, response, generated_text


    def get_text_query(self, image_bytes:bytes, predefined_prompt:str) -> dict:
        prompt = get_prompt(predefined_prompt)
        response = self.gemini.generate_content([G_Image.from_bytes(image_bytes), prompt])

        structured_output = extract_json(response.text)
        if len(structured_output):
            return structured_output[0]
        else:
            raise ValueError(f"")


    def edit_image_mask_free(self, 
                             image, 
                             prompt:str, 
                             negative_prompt:str, 
                             guidance_scale:int=21, 
                             number_of_images:int=4) -> vertexai.preview.vision_models.ImageGenerationResponse:
        
        from vertexai.preview.vision_models import Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@002")
        base_img = Image(image)
        
        images = model.edit_image(
            base_image=base_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            # Optional parameters
            seed=1,
            # Controls the strength of the prompt.
            # -- 0-9 (low strength), 10-20 (medium strength), 21+ (high strength)
            guidance_scale=guidance_scale,
            number_of_images=number_of_images,
        )
    
        return images

    
    def edit_image_mask(self,
                        image,
                        mask,
                        prompt: str, 
                        negative_prompt: str,
                        guidance_scale:int=21,
                        number_of_images:int=4) -> vertexai.preview.vision_models.ImageGenerationResponse:
    
        from vertexai.preview.vision_models import Image
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        base_img = Image(image)
        mask_img = Image(mask)
    
        images = model.edit_image(
            base_image=base_img,
            mask=mask_img,
            prompt=prompt,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            number_of_images=number_of_images,
        )
    
        return images
                            
        
    def stable_diffusion_xl(self,
                            project: str,
                            endpoint_id: str,
                            instances: Union[Dict, List[Dict]],
                            location: str = "asia-northeast3",
                            api_endpoint: str = "asia-northeast3-aiplatform.googleapis.com"):
        """
        `instances` can be either single instance of type dict or a list
        of instances.
        """
        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": api_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = instances if isinstance(instances, list) else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        print("response")
        print(" deployed_model_id:", response.deployed_model_id)
        # The predictions are a google.protobuf.Value representation of the model's predictions.
        predictions = response.predictions
        #for prediction in predictions:
        #    print(" prediction:", dict(prediction))
    
        return predictions
