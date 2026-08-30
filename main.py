import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    reload = os.getenv("APP_ENV") == "development"
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, reload=reload)


if __name__ == "__main__":
    main()
