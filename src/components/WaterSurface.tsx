import {
  AlphaType,
  Canvas,
  ColorType,
  Fill,
  FilterMode,
  ImageShader,
  MipmapMode,
  Shader,
  Skia,
  useImage,
} from '@shopify/react-native-skia';
import { useMemo, useRef } from 'react';
import { StyleSheet, View, useWindowDimensions } from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useFrameCallback, useSharedValue } from 'react-native-reanimated';

// ---------------------------------------------------------------------------
// TUNABLE PARAMETERS
// All of the "feel" of the water lives here. Austen will fine-tune on-device.
// ---------------------------------------------------------------------------

/** Width of the simulation grid in cells. Height is derived from screen aspect.
 *  Higher = finer ripples but more CPU per frame. 120 is a good 60fps target. */
const GRID_WIDTH = 120;

/** Energy retained each step (0..1). Lower = ripples die faster / calmer pool.
 *  0.985 gives a glassy surface that settles over ~2s. */
const DAMPING = 0.985;

/** Height pushed into the water under the finger, per stamp (sim units).
 *  Negative = the finger dents the surface downward like a real poke. */
const IMPULSE_STRENGTH = -6.0;

/** Radius of the finger "brush" in grid cells. Larger = fatter wake. */
const BRUSH_RADIUS = 3.5;

/** Maps raw sim height into the 0..1 range stored in the height texture.
 *  Keep IMPULSE_STRENGTH * HEIGHT_SCALE comfortably inside ~0.45 so it
 *  doesn't clip the encoding. */
const HEIGHT_SCALE = 0.07;

/** How far (in screen px) the surface gradient bends the image beneath.
 *  This is the strength of the refraction. */
const REFRACTION_STRENGTH = 70.0;

/** Screen-px radius of the cheap disturbance blur where the water is choppy. */
const BLUR_STRENGTH = 4.0;

/** Strength of the subtle specular glint on wave slopes. */
const HIGHLIGHT_STRENGTH = 0.25;

/** How much choppy water darkens the image (0..1). */
const SHADING_STRENGTH = 0.35;

// ---------------------------------------------------------------------------
// SkSL runtime shader: refracts the background using the height field.
// Child shader 0 = background (cover-fit to the screen).
// Child shader 1 = height field (sampled in its own grid-pixel space).
// ---------------------------------------------------------------------------

const source = Skia.RuntimeEffect.Make(`
uniform shader image;      // background, cover-fit to the screen (canvas space)
uniform shader heightMap;  // height field, sampled in grid-pixel space
uniform float2 resolution; // screen size in px
uniform float2 gridSize;   // grid dimensions in cells
uniform float refraction;  // px offset per unit gradient
uniform float blurAmount;  // px blur radius at max disturbance
uniform float highlight;   // specular strength
uniform float shading;     // darkening strength

// Decode the height stored (biased around 0.5) in the red channel.
float sampleH(float2 g) {
  return heightMap.eval(g).r - 0.5;
}

half4 main(float2 xy) {
  // Map this screen pixel into grid space.
  float2 g = xy / resolution * gridSize;

  // Central-difference gradient of the surface (one cell each way).
  float hL = sampleH(g - float2(1.0, 0.0));
  float hR = sampleH(g + float2(1.0, 0.0));
  float hU = sampleH(g - float2(0.0, 1.0));
  float hD = sampleH(g + float2(0.0, 1.0));
  float2 grad = float2(hR - hL, hD - hU);

  // Refract: sample the image at an offset proportional to the slope.
  float2 sampleXY = xy + grad * refraction;

  // How disturbed is the water right here (0 = glassy, 1 = choppy).
  float disturb = clamp(length(grad) * 8.0, 0.0, 1.0);

  half4 col = image.eval(sampleXY);

  // Cheap blur that only kicks in on choppy water: a 4-tap ring blended by
  // the disturbance amount so calm water stays crisp.
  if (blurAmount > 0.0) {
    float r = blurAmount * disturb;
    half4 ring = image.eval(sampleXY + float2(r, 0.0))
               + image.eval(sampleXY - float2(r, 0.0))
               + image.eval(sampleXY + float2(0.0, r))
               + image.eval(sampleXY - float2(0.0, r));
    col = mix(col, (col + ring) / 5.0, disturb);
  }

  // Subtle specular glint from slopes facing a virtual light (top-left).
  float spec = clamp((-grad.x - grad.y) * 0.5, 0.0, 1.0);
  col.rgb += spec * highlight;

  // Choppy water reads slightly darker, like a broken reflection.
  col.rgb *= (1.0 - disturb * shading);

  return col;
}
`);

export function WaterSurface() {
  const { width, height } = useWindowDimensions();

  const background = useImage(require('@/assets/images/background.jpg'));

  // Grid dimensions: keep cells roughly square by matching screen aspect.
  const gridW = GRID_WIDTH;
  const gridH = Math.max(2, Math.round(GRID_WIDTH * (height / width)));
  const cellCount = gridW * gridH;

  // Simulation state lives in a ref that is only ever touched on the UI thread
  // inside the frame worklet (never during render). Refs are the mutable escape
  // hatch the React Compiler allows, and keeping all access inside the worklet
  // keeps the "no refs during render" rule happy too. The buffers are allocated
  // lazily on the first frame and reused (mutated in place) every frame after.
  const buffersRef = useRef<{
    current: Float32Array;
    previous: Float32Array;
    rgba: Uint8Array;
  } | null>(null);

  // A flat (undisturbed) height image so the shader always has a valid child.
  const flatImage = useMemo(() => {
    const rgba = new Uint8Array(cellCount * 4);
    for (let i = 0; i < cellCount; i++) {
      rgba[i * 4] = 128; // 0.5 -> zero height
      rgba[i * 4 + 3] = 255;
    }
    const data = Skia.Data.fromBytes(rgba);
    return Skia.Image.MakeImage(
      { width: gridW, height: gridH, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Opaque },
      data,
      gridW * 4,
    );
  }, [cellCount, gridW, gridH]);

  // The per-frame height texture handed to the shader.
  const heightImage = useSharedValue(flatImage);

  // Touch bridge (gesture worklets -> frame worklet, all on the UI thread).
  const touchX = useSharedValue(0);
  const touchY = useSharedValue(0);
  const prevX = useSharedValue(0);
  const prevY = useSharedValue(0);
  const touching = useSharedValue(0);

  const uniforms = useMemo(
    () => ({
      resolution: [width, height],
      gridSize: [gridW, gridH],
      refraction: REFRACTION_STRENGTH,
      blurAmount: BLUR_STRENGTH,
      highlight: HIGHLIGHT_STRENGTH,
      shading: SHADING_STRENGTH,
    }),
    [width, height, gridW, gridH],
  );

  const fullScreenRect = useMemo(
    () => ({ x: 0, y: 0, width, height }),
    [width, height],
  );

  const pan = Gesture.Pan()
    .minDistance(0)
    .onBegin((e) => {
      'worklet';
      prevX.value = e.x;
      prevY.value = e.y;
      touchX.value = e.x;
      touchY.value = e.y;
      touching.value = 1;
    })
    .onUpdate((e) => {
      'worklet';
      touchX.value = e.x;
      touchY.value = e.y;
    })
    .onFinalize(() => {
      'worklet';
      touching.value = 0;
    });

  useFrameCallback(() => {
    'worklet';
    // Lazily allocate (or resize) the simulation buffers on the UI thread.
    if (buffersRef.current === null || buffersRef.current.rgba.length !== cellCount * 4) {
      buffersRef.current = {
        current: new Float32Array(cellCount),
        previous: new Float32Array(cellCount),
        rgba: new Uint8Array(cellCount * 4),
      };
    }
    const buffers = buffersRef.current;
    const current = buffers.current;
    const previous = buffers.previous;
    const rgba = buffers.rgba;

    // --- 1. Inject the finger impulse along the drag segment (continuous wake).
    if (touching.value === 1) {
      const cxGrid = (touchX.value / width) * gridW;
      const cyGrid = (touchY.value / height) * gridH;
      const pxGrid = (prevX.value / width) * gridW;
      const pyGrid = (prevY.value / height) * gridH;

      const dx = cxGrid - pxGrid;
      const dy = cyGrid - pyGrid;
      const segLen = Math.sqrt(dx * dx + dy * dy);
      const stamps = Math.max(1, Math.ceil(segLen));

      const r2 = BRUSH_RADIUS * BRUSH_RADIUS;
      const rCeil = Math.ceil(BRUSH_RADIUS);

      for (let s = 0; s < stamps; s++) {
        const t = stamps === 1 ? 0 : s / (stamps - 1);
        const sx = pxGrid + dx * t;
        const sy = pyGrid + dy * t;
        const minX = Math.max(1, Math.floor(sx) - rCeil);
        const maxX = Math.min(gridW - 2, Math.floor(sx) + rCeil);
        const minY = Math.max(1, Math.floor(sy) - rCeil);
        const maxY = Math.min(gridH - 2, Math.floor(sy) + rCeil);
        for (let y = minY; y <= maxY; y++) {
          for (let x = minX; x <= maxX; x++) {
            const ox = x - sx;
            const oy = y - sy;
            const d2 = ox * ox + oy * oy;
            if (d2 <= r2) {
              // Smooth falloff so the brush edge isn't a hard disc.
              const falloff = 1 - d2 / r2;
              current[y * gridW + x] += IMPULSE_STRENGTH * falloff;
            }
          }
        }
      }

      prevX.value = touchX.value;
      prevY.value = touchY.value;
    }

    // --- 2. Wave-equation step (Hugo Elias scheme). Borders stay 0 -> reflect.
    for (let y = 1; y < gridH - 1; y++) {
      const row = y * gridW;
      for (let x = 1; x < gridW - 1; x++) {
        const i = row + x;
        const next =
          ((current[i - 1] + current[i + 1] + current[i - gridW] + current[i + gridW]) * 0.5 -
            previous[i]) *
          DAMPING;
        previous[i] = next;
      }
    }
    // Swap: `previous` now holds the newest field.
    buffers.current = previous;
    buffers.previous = current;

    // --- 3. Encode the newest field into the RGBA texture.
    const field = buffers.current;
    for (let i = 0; i < cellCount; i++) {
      let v = 0.5 + field[i] * HEIGHT_SCALE;
      if (v < 0) v = 0;
      else if (v > 1) v = 1;
      rgba[i * 4] = (v * 255) | 0;
      rgba[i * 4 + 3] = 255;
    }

    // --- 4. Hand the height field to the shader as a fresh Skia image.
    const data = Skia.Data.fromBytes(rgba);
    const img = Skia.Image.MakeImage(
      { width: gridW, height: gridH, colorType: ColorType.RGBA_8888, alphaType: AlphaType.Opaque },
      data,
      gridW * 4,
    );
    if (img) {
      heightImage.value = img;
    }
  });

  // Wait for the background to decode and the shader to compile before drawing
  // so the shader never samples a null child.
  if (!background || !source) {
    return <View style={styles.placeholder} />;
  }

  return (
    <GestureDetector gesture={pan}>
      <Canvas style={styles.canvas}>
        <Fill>
          <Shader source={source} uniforms={uniforms}>
            <ImageShader
              image={background}
              fit="cover"
              rect={fullScreenRect}
              tx="clamp"
              ty="clamp"
            />
            <ImageShader
              image={heightImage}
              fit="none"
              tx="clamp"
              ty="clamp"
              sampling={{ filter: FilterMode.Linear, mipmap: MipmapMode.None }}
            />
          </Shader>
        </Fill>
      </Canvas>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  canvas: {
    flex: 1,
  },
  placeholder: {
    flex: 1,
    backgroundColor: '#000',
  },
});
