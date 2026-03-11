"""Entry point: python -m mental_models_mcp"""

import asyncio
from .server import main

asyncio.run(main())
