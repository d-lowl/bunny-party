from typing import Dict, Any, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from server.embedding import TerminatorEmbedding
from server.model import run_model_training_experiment, get_model_evaluation

app = FastAPI()
embedding = TerminatorEmbedding("train")


@app.post("/train_model")
def train_model() -> None:
    """Run the model training experiment."""
    run_model_training_experiment()
    return


@app.get("/evaluate_model")
def evaluate_model() -> Dict[str, float]:
    """Evaluate the current model using the test dataset.

    Returns:
        dict: of evaluation metrics
    """
    return get_model_evaluation()


@app.get("/similar_images")
def get_similar_images(img_id: str, class_id: str, n: int = 3) -> List[Dict[str, Any]]:
    """Get images similar to the given

    Args:
        img_id (str): image ID to search for
        class_id (str): class label ID
        n (int): number of images to return back

    Returns:
        list: of dictionaries describing the similar images
    """
    similar_images = embedding.get_n_most_similar(img_id, n, class_id)
    return similar_images.drop("embedding", axis=1).to_dict(orient="records")


class RelabelRequest(BaseModel):
    """Relabel request data."""
    img_id: str
    class_id: str


@app.post("/relabel")
def relabel(request: RelabelRequest) -> None:
    """Relabeling endpoint."""
    embedding.relabel(img_id=request.img_id, class_id=request.class_id)



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
