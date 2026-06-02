from pydantic import BaseModel, ConfigDict


class AppBaseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class PaginationParams(AppBaseModel):
    skip: int = 0
    limit: int = 20
