from pydantic import BaseModel, ConfigDict, Field


class UsedCarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    year: int = Field(ge=1886, le=2100)
    km_driven: float = Field(ge=0, allow_inf_nan=False)
    fuel: str = Field(min_length=1, max_length=50)
    seller_type: str = Field(min_length=1, max_length=80)
    transmission: str = Field(min_length=1, max_length=50)
    owner: str = Field(min_length=1, max_length=80)


class PredictionResponse(BaseModel):
    predicted_selling_price: float
    currency: str = "INR"
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool