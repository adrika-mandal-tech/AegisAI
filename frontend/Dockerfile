# frontend/Dockerfile
# Build with: docker compose -f docker/docker-compose.yml up --build frontend

FROM node:22-alpine AS base

# 1. Install dependencies
FROM base AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

# 2. Build the app
FROM base AS builder
WORKDIR /app
ARG AEGIS_BACKEND_URL=http://localhost:8000
ARG NEXT_PUBLIC_API_URL=/api
ENV AEGIS_BACKEND_URL=${AEGIS_BACKEND_URL}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .
RUN mkdir -p /app/public
RUN npm run build

# 3. Final production image
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
COPY --from=builder --chown=node:node /app/public ./public
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
USER node
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
