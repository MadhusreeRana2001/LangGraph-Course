from pydantic import BaseModel, Field
from typing import Annotated, Union, Any


class TypeAnnotations(BaseModel):
    x: Annotated[
        Union[int, float],
        Field(description="Any int or float number")
    ]
    y: Annotated[
        Any,
        Field(description="Any data")
    ]


typeAnnotations_1 = TypeAnnotations(x = 2, y = "Hello World")
print(typeAnnotations_1)  # x=2 y='Hello World'

typeAnnotations_2 = TypeAnnotations(x = 2.5, y = {"name": "John"})
print(typeAnnotations_2)  # x=2.5 y={'name': 'John'}