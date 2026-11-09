"""Development only. Production schema should be migrated using reviewed migrations."""
from .db import Base, engine
from . import models  # noqa
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Development tables created")
