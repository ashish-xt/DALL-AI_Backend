FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy source files
COPY server.js ./

# Create tmp directory for TTS audio generation
RUN mkdir -p /tmp

EXPOSE 5000

CMD ["node", "server.js"]
