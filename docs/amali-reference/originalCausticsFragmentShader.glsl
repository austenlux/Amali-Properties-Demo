
        precision highp float;

        varying vec2 vTextureCoord;

        uniform sampler2D uSampler0;
        uniform sampler2D uRippleTexture;
        uniform float uTime;
        uniform vec2 uMousePosition;
        uniform vec2 uPreviousMousePos;
        uniform float uMouseCircleOpacity;
        uniform float uDebugMode;

        // Exact Unicorn Studio caustics shader code
        vec4 permute(vec4 t) { return t * (t * 34.0 + 133.0); }

        vec3 grad(float hash) {
            vec3 cube = mod(floor(hash / vec3(1.0, 2.0, 4.0)), 2.0) * 2.0 - 1.0;
            vec3 cuboct = cube;
            float index0 = step(0.0, 1.0 - floor(hash / 16.0));
            float index1 = step(0.0, floor(hash / 16.0) - 1.0);
            cuboct.x *= 1.0 - index0;
            cuboct.y *= 1.0 - index1;
            cuboct.z *= 1.0 - (1.0 - index0 - index1);
            float type = mod(floor(hash / 8.0), 2.0);
            vec3 rhomb = (1.0 - type) * cube + type * (cuboct + cross(cube, cuboct));
            vec3 grad = cuboct * 1.22474487139 + rhomb;
            grad *= (1.0 - 0.042942436724648037 * type) * 3.5946317686139184;
            return grad;
        }

        vec4 bccNoiseDerivativesPart(vec3 X) {
            vec3 b = floor(X);
            vec4 i4 = vec4(X - b, 2.5);
            vec3 v1 = b + floor(dot(i4, vec4(.25)));
            vec3 v2 = b + vec3(1, 0, 0) + vec3(-1, 1, 1) * floor(dot(i4, vec4(-.25, .25, .25, .35)));
            vec3 v3 = b + vec3(0, 1, 0) + vec3(1, -1, 1) * floor(dot(i4, vec4(.25, -.25, .25, .35)));
            vec3 v4 = b + vec3(0, 0, 1) + vec3(1, 1, -1) * floor(dot(i4, vec4(.25, .25, -.25, .35)));
            vec4 hashes = permute(mod(vec4(v1.x, v2.x, v3.x, v4.x), 289.0));
            hashes = permute(mod(hashes + vec4(v1.y, v2.y, v3.y, v4.y), 289.0));
            hashes = mod(permute(mod(hashes + vec4(v1.z, v2.z, v3.z, v4.z), 289.0)), 48.0);
            vec3 d1 = X - v1; vec3 d2 = X - v2; vec3 d3 = X - v3; vec3 d4 = X - v4;
            vec4 a = max(0.75 - vec4(dot(d1, d1), dot(d2, d2), dot(d3, d3), dot(d4, d4)), 0.0);
            vec4 aa = a * a; vec4 aaaa = aa * aa;
            vec3 g1 = grad(hashes.x); vec3 g2 = grad(hashes.y); vec3 g3 = grad(hashes.z); vec3 g4 = grad(hashes.w);
            vec4 extrapolations = vec4(dot(d1, g1), dot(d2, g2), dot(d3, g3), dot(d4, g4));
            vec3 derivative = -8.0 * (d1 * (aa.x * a.x * extrapolations.x) + d2 * (aa.y * a.y * extrapolations.y) + d3 * (aa.z * a.z * extrapolations.z) + d4 * (aa.w * a.w * extrapolations.w)) + (g1 * aaaa.x + g2 * aaaa.y + g3 * aaaa.z + g4 * aaaa.w);
            return vec4(derivative, dot(aaaa, extrapolations));
        }

        vec4 bccNoiseDerivatives_XYBeforeZ(vec3 X) {
            mat3 orthonormalMap = mat3(
                0.788675134594813, -0.211324865405187, -0.577350269189626,
                -0.211324865405187, 0.788675134594813, -0.577350269189626,
                0.577350269189626, 0.577350269189626, 0.577350269189626
            );
            X = orthonormalMap * X;
            vec4 result = bccNoiseDerivativesPart(X) + bccNoiseDerivativesPart(X + 144.5);
            return vec4(result.xyz * orthonormalMap, result.w);
        }

        const float PI = 3.14159265359;

        vec4 normalizeNoise(vec4 noise, float amount) {
            return mix(noise, (noise + 0.5) * 0.5, amount);
        }

        mat2 rotate2d(float angle) {
            return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
        }

        vec4 getNoise(vec3 p) {
            vec4 noise = bccNoiseDerivatives_XYBeforeZ(p);
            return normalizeNoise(noise, 0.0);
        }

        vec3 blend(vec3 src, vec3 dst) {
            return 1.0 - (1.0 - src) * (1.0 - dst);
        }

        void getCaustics(vec2 uv, out vec4 outNoise, out vec3 outColor) {
            vec2 aspect = vec2(1.0, 1.0);
            vec2 mPos = vec2(0.14372822299651566, 0.03765243902439008) + mix(vec2(0), (uMousePosition-0.5), 0.0);
            float mDist = max(0.0,1.0 - distance(uv * aspect, mPos * aspect) * 4.0 * (1.0 - 0.81));

            uv -= vec2(0.14372822299651566, 0.03765243902439008);
            uv = uv * aspect * rotate2d(-0.0189 * 2.0 * PI) * vec2(1.0, 1.0) * 16.0 * 0.446;

            float refraction = mix(0.25, 1.3, 0.95);
            vec3 p = vec3(uv, uTime * 0.25);
            vec4 noise = getNoise(p);
            vec4 baseNoise = noise;
            vec4 balanceNoise = getNoise(p - vec3(baseNoise.xyz / 32.0) * refraction);
            noise = getNoise(p - vec3(balanceNoise.xyz / 16.0) * refraction);

            float balancer = (0.5 + 0.5 * balanceNoise.w);
            float normalized = pow(0.5 + 0.5 * noise.w, 2.0);
            float value = mix(0.0, normalized + 0.2 * (1.0 - normalized), balancer * mDist);
            outNoise = baseNoise * mDist;
            outColor = vec3(0.4196078431372549, 0.8235294117647058, 1.0) * value;
        }

        void main() {
            vec2 uv = vTextureCoord;
            
            // Apply caustics only
            vec4 causticNoise;
            vec3 causticColorOut;
            getCaustics(uv, causticNoise, causticColorOut);
            
            // Read ripple data and combine with caustic distortion
            vec4 rippleData = texture2D(uRippleTexture, uv);
            float height = rippleData.r;
            
            // Debug mode: show ripple data directly
            if (uDebugMode > 0.5) {
                vec3 debugColor = vec3(abs(height) * 20.0, abs(rippleData.g) * 100.0, 0.0);
                gl_FragColor = vec4(clamp(debugColor, 0.0, 1.0), 1.0);
                return;
            }
            
            // Combine caustic and ripple distortions
            vec2 rippleDistortion = vec2(height * 0.02);
            vec2 totalDistortion = causticNoise.xy * 0.01 * 0.25 + rippleDistortion;
            
            // Apply chromatic aberration to ripple distortion (simple approach)
            // Increased strength to make the rainbow effect more visible
            vec2 chromaticOffset = rippleDistortion * 0.2; // Much stronger chromatic aberration
            vec4 leftColor = texture2D(uSampler0, uv + totalDistortion - chromaticOffset);
            vec4 rightColor = texture2D(uSampler0, uv + totalDistortion + chromaticOffset);
            
            // Sample with caustic distortion
            vec4 color = texture2D(uSampler0, uv + totalDistortion);
            
            // Apply ripple chromatic aberration (more dramatic effect)
            if (length(rippleDistortion) > 0.001) {
                color.r = leftColor.r;
                color.b = rightColor.b;
                // Also add some green channel distortion for more rainbow effect
                color.g = mix(color.g, leftColor.g, 0.3);
                
                // Debug: Add a subtle color tint to make chromatic aberration more visible
                if (uDebugMode > 0.5) {
                    color.rgb += vec3(0.1, 0.0, 0.1) * length(rippleDistortion) * 10.0; // Red/blue tint
                }
            }
            
            // Apply existing chromatic aberration system
            float aspectRatio = 1.0;
            vec2 mPos = vec2(0.48519163763066203, 0.5111498257839722) + mix(vec2(0), (uMousePosition-0.5), 0.0);
            vec2 pos = vec2(0.48519163763066203, 0.5111498257839722);
            
            float mDist = max(0.0, 1.0 - distance(uv * vec2(aspectRatio, 1.0), mPos * vec2(aspectRatio, 1.0)) * 4.0 * (1.0 - 1.0));
            vec2 aberrated;
            vec2 dir = uv - pos;
            float dist = length(dir);
            dir = normalize(dir);
            aberrated = 0.3 * dir * 0.015 * mix(1.0, dist * (1.0 + 0.4), 0.4);
            aberrated *= mDist;

            float amt = length(aberrated);
            // Chromatic aberration code - hidden for now but kept for water ripple implementation
            /*
            if(amt >= 0.001) {
                vec4 left = vec4(0.0);
                vec4 right = vec4(0.0);
                vec4 center = vec4(0.0);
                float invSteps = 1.0 / 15.0;

                for (int i = 0; i <= 14; i++) {
                    float fi = float(i);
                    vec2 offset = aberrated * (fi * invSteps);
                    left += texture2D(uSampler0, uv - offset + totalDistortion) * invSteps;
                    right += texture2D(uSampler0, uv + offset + totalDistortion) * invSteps;
                }

                for (int i = 0; i <= 14; i++) {
                    float fi = float(i);
                    vec2 offset = aberrated * ((fi / 14.0) - 0.5);
                    center += texture2D(uSampler0, uv + offset + totalDistortion) * invSteps;
                }

                color.r = left.r;
                color.g = mix(color.g, center.g, 1.0);
                color.b = right.b;
                color.a = max(max(left.a, center.a), right.a);
            }
            */
            
            // Blend caustics
            vec3 blended = blend(color.rgb, causticColorOut);
            color.rgb = mix(color.rgb, blended, 0.74);

            gl_FragColor = color;
        }
    