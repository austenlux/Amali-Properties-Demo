
        precision highp float;

        varying vec3 vVertexPosition;
        varying vec2 vTextureCoord;

        uniform sampler2D uTexture;
        uniform sampler2D uPingPongTexture;
        uniform vec2 uPreviousMousePos;
        uniform vec2 uMousePos;
        uniform vec2 uResolution;
        uniform float uRadius;
        uniform float uDissipate;
        uniform float uDecay;
        uniform float uScale;
        uniform float uLiquidity;
        uniform float uSpeed;

        const float PI = 3.1415926;
        const float TWOPI = 6.2831852;

        void main() {
            vec2 aspect = vec2(uResolution.x/uResolution.y, 1.0);
            vec2 texelSize = (1.0 / (vec2(1080.0) * aspect)) * mix(1.0, 8.0, uSpeed);
            vec2 vUv = vTextureCoord;
            // When scale >= 1.0, use full canvas coordinates; otherwise use Unicorn Studio's scaling
            vec2 mPos = (uScale >= 1.0) ? uMousePos : mix(uMousePos, (uMousePos - 0.5) * 0.5 + 0.5, uScale);
            vec2 pmPos = (uScale >= 1.0) ? uPreviousMousePos : mix(uPreviousMousePos, (uPreviousMousePos - 0.5) * 0.5 + 0.5, uScale);

            float waveSpeed = 1.0;
            float damping = mix(0.8, 0.999, uDecay);
            float velocityDamping = damping;
            float heightDamping = damping;
            float time = 0.5;

            vec4 data = texture2D(uPingPongTexture, vUv);
            float height = data.r;
            float velocity = data.g;

            float laplacian = 0.0;
            float totalWeight = 0.0;
            float scaleDiff = uScale * 0.25;
            // Clamp regions for wave propagation - when scale >= 1.0, use full canvas
            vec2 clampRegionMin = vec2(max(0.0, uScale * 0.5 - scaleDiff));
            vec2 clampRegionMax = vec2(min(1.0, 1.0 - uScale * 0.5 + scaleDiff));

            // Right sample
            vec2 offset = vec2(texelSize.x, 0.0);
            vec2 neighborUv = (uScale >= 1.0) ? clamp(vUv + offset, vec2(0.0), vec2(1.0)) : clamp(vUv + offset, clampRegionMin, clampRegionMax);
            float weight = 1.0 - length(offset) / (length(texelSize) * 2.0);
            laplacian += texture2D(uPingPongTexture, neighborUv).r * weight;
            totalWeight += weight;

            // Left sample
            offset = vec2(-texelSize.x, 0.0);
            neighborUv = (uScale >= 1.0) ? clamp(vUv + offset, vec2(0.0), vec2(1.0)) : clamp(vUv + offset, clampRegionMin, clampRegionMax);
            weight = 1.0 - length(offset) / (length(texelSize) * 2.0);
            laplacian += texture2D(uPingPongTexture, neighborUv).r * weight;
            totalWeight += weight;

            // Up sample
            offset = vec2(0.0, texelSize.y);
            neighborUv = (uScale >= 1.0) ? clamp(vUv + offset, vec2(0.0), vec2(1.0)) : clamp(vUv + offset, clampRegionMin, clampRegionMax);
            weight = 1.0 - length(offset) / (length(texelSize) * 2.0);
            laplacian += texture2D(uPingPongTexture, neighborUv).r * weight;
            totalWeight += weight;

            // Down sample
            offset = vec2(0.0, -texelSize.y);
            neighborUv = (uScale >= 1.0) ? clamp(vUv + offset, vec2(0.0), vec2(1.0)) : clamp(vUv + offset, clampRegionMin, clampRegionMax);
            weight = 1.0 - length(offset) / (length(texelSize) * 2.0);
            laplacian += texture2D(uPingPongTexture, neighborUv).r * weight;
            totalWeight += weight;

            float avgNeighbors = laplacian / totalWeight;
            laplacian = avgNeighbors - height;

            velocity += waveSpeed * waveSpeed * laplacian;
            velocity *= velocityDamping;
            
            height += velocity;
            height *= heightDamping;

            float mouseSpeed = distance(mPos * aspect, pmPos * aspect);
            float dist = distance(vUv * aspect, mPos * aspect);
            // Scale affects ripple radius - larger scale = larger ripples
            float radius = 0.025 * uScale;
            
            if (dist < radius && mouseSpeed > 0.0001) {
                float drop = cos(dist / radius * PI * time);
                float intensity = mouseSpeed * 20.0;
                height += drop * intensity;
            }

            gl_FragColor = vec4(height, velocity, 0.0, 1.0);
        }
    