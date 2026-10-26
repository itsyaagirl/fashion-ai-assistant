"""Typed contracts between the LLM and the rest of the code."""
from __future__ import annotations

from pydantic import BaseModel, Field


class OutfitAnalysis(BaseModel):
    """What the vision model sees in the outfit photo."""

    is_outfit: bool = Field(description="True if the photo shows clothing / an outfit (worn or laid flat).")
    garments: list[str] = Field(default_factory=list, description="Visible clothing items, e.g. 'black wide-leg trousers'.")
    colors: list[str] = Field(default_factory=list, description="2-4 main colours of the outfit as simple names, e.g. 'navy', 'cream'.")
    style: str = Field(default="casual", description="One of: minimalist, streetwear, classic, luxe, boho, sporty, casual, formal, romantic, edgy.")
    occasion: str = Field(default="everyday", description="Most likely occasion, e.g. office, date night, weekend, wedding guest, festival, gym.")
    season: str = Field(default="all-season", description="spring, summer, autumn, winter or all-season.")
    formality: int = Field(default=2, ge=1, le=5, description="1 = very casual, 5 = black tie.")
    existing_accessories: list[str] = Field(default_factory=list, description="Accessories already worn in the photo.")
    suggested_categories: list[str] = Field(default_factory=list,
                                            description="2-4 accessory categories that would complete the look, e.g. 'earrings', 'bag', 'belt'.")
    vibe: str = Field(default="", description="One sentence describing the overall look.")
