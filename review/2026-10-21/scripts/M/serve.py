import os, sys
from http.server import ThreadingHTTPServer
from eAMFModel import trader
W = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/M/work"
os.environ.pop("EAMF_V21_DRIVES", None)
paths = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
t = trader.V21Trader(W, paths=paths, prior=os.path.join(W, "avg_form4_w05.csv"))
srv = ThreadingHTTPServer(("127.0.0.1", 18751), trader.handler_for(t))
print("serving", flush=True)
srv.serve_forever()
