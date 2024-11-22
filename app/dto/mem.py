from typing import List, Optional

from pydantic import BaseModel


class AddPayload(BaseModel):
    messages: List[dict[str, str]]
    user_name: str
    namespace: str
    metadata: Optional[dict] = None
    filters: Optional[dict] = None