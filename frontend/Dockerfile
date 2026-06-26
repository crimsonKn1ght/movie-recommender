# Frontend image: build the CRA app, then serve the static bundle with nginx.
# nginx also reverse-proxies /api/* to the backend service (see nginx.conf).
# Build context must be the repository root:
#   docker build -f frontend/Dockerfile -t movie-recommender-frontend .

# --- build stage ---
FROM node:18-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- serve stage ---
FROM nginx:alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
