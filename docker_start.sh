docker run -i \
  -v ~/.nanobot:/root/.nanobot \
  -v ~/sim/:/sim \
  -v ~/sim/prediction/bk_quinlan_stock/data/cache_output:/tmp/cache_output \
  -v ~/sim/prediction/bk_quinlan_stock/data/candidates:/tmp/candidates \
  -p 18790:18790 nanobot:202602 gateway
