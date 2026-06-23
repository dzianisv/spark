# spark + Aurora unified sandbox image
# Aurora (ChatGPT proxy, Go): github.com/aurora-develop/aurora
# Spark runtime: oven/bun:alpine

FROM oven/bun:1-alpine AS base

# Install Go + build tools needed for `go install`
RUN apk add --no-cache go git ca-certificates curl

# Install Aurora via go install
ENV GOPATH=/root/go
ENV PATH=$GOPATH/bin:$PATH
RUN go install github.com/aurora-develop/aurora@latest

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bun", "/spark/spark.ts"]
