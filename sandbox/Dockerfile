FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    curl git ca-certificates unzip xz-utils \
    build-essential procps \
    && rm -rf /var/lib/apt/lists/*

# Node.js 22 + npm
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Bun
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"

# Go
ENV GO_VERSION=1.24.4
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/').tar.gz" \
    | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:/root/go/bin:$PATH"
ENV GOPATH=/root/go

# Aurora — build into image so startup is instant
RUN go install github.com/aurora-develop/aurora@latest

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
