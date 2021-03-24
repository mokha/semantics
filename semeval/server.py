import uvicorn
import argparse
from semeval.serviecs import embeddings

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=None, help="Hostname (default: localhost)")
    p.add_argument("--port", default=None, help="Port (default: 1337)")
    p.add_argument("--service", required=True, help="The service to run (e.g. embeddings)")
    args = p.parse_args()

    services = {
        'embeddings': embeddings.EmbeddingsServer,
    }

    if args.service not in services:
        raise Exception("Service '{}' not available!".format(args.service))

    uvicorn.run(services[args.service](**args.__dict__),
                host=args.host if args.host else "localhost",
                port=int(args.port) if args.port else 1337)
