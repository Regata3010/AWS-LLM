# deployment/frontend/Dockerfile
# Build from repo root: docker build -f deployment/frontend/Dockerfile -t biasguard-frontend ./frontend

# ============================================
# Stage 1: Builder (Build React app)
# ============================================
FROM node:20.18-alpine AS builder

WORKDIR /build

# Copy package files first (better caching)
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --production=false --silent

# Copy ALL source code (respects ./frontend/.dockerignore)
COPY . ./

# Build production bundle
RUN npm run build

# Verify dist exists
RUN ls -la /build/dist || (echo "ERROR: dist folder not created!" && exit 1)

# ============================================
# Stage 2: Runtime (Serve with Nginx)
# ============================================
FROM nginx:alpine

# Remove default nginx config
RUN rm -f /etc/nginx/conf.d/default.conf

# Create app directory first
RUN mkdir -p /usr/share/nginx/html

# Copy built React app from builder stage
COPY --from=builder /build/dist/ /usr/share/nginx/html/

# Copy nginx config (must exist in ./frontend/)
# Make sure nginx.conf exists in frontend/ folder!
COPY ./nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user
RUN addgroup -g 1000 biasguard && \
    adduser -D -u 1000 -G biasguard biasguard && \
    chown -R biasguard:biasguard /usr/share/nginx/html && \
    chown -R biasguard:biasguard /var/cache/nginx && \
    chown -R biasguard:biasguard /var/log/nginx && \
    chown -R biasguard:biasguard /etc/nginx/conf.d && \
    touch /var/run/nginx.pid && \
    chown biasguard:biasguard /var/run/nginx.pid

# Switch to non-root user
USER biasguard

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]