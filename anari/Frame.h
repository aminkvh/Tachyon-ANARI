// Copyright 2021-2026 The Khronos Group
// Copyright (c) 2026 Tachyon-dev contributors (adapted from the helide device)
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"
#include "Renderer.h"
#include "TachyonScene.h"
#include "World.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <vector>

namespace tachyon_anari {

struct Frame : public helium::BaseFrame
{
  Frame(TachyonGlobalState *s);
  ~Frame() override = default;

  bool isValid() const override;

  TachyonGlobalState *deviceState() const;

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  void finalize() override;

  void renderFrame() override;

  void *map(std::string_view channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;
  void unmap(std::string_view channel) override;
  int frameReady(ANARIWaitMask m) override;
  void discard() override;

 private:
  void rebuildScene(helium::TimeStamp sceneStamp);
  void writePixels();
  void fillWithBackground();

  uint2 m_size{0u, 0u};
  uint2 m_incomingSize{0u, 0u};
  ANARIDataType m_colorType{ANARI_UNKNOWN};
  ANARIDataType m_depthType{ANARI_UNKNOWN};
  ANARIDataType m_incomingColorType{ANARI_UNKNOWN};
  ANARIDataType m_incomingDepthType{ANARI_UNKNOWN};

  std::vector<uint8_t> m_pixelBuffer;
  std::vector<float> m_depthBuffer;
  std::vector<float> m_depthRaw; // Tachyon hit distances (1e18: miss)
  std::vector<float> m_rgb; // Tachyon's float output, 3 floats per pixel

  helium::IntrusivePtr<Renderer> m_renderer;
  helium::IntrusivePtr<Camera> m_camera;
  helium::IntrusivePtr<World> m_world;

  // Change tracking: what the current Tachyon scene was built from
  TachyonScene m_scene;
  bool m_sceneValid{false};
  bool m_lit{false};
  bool m_builtEyeLight{false};
  helium::TimeStamp m_builtStamp{0};
  const World *m_builtWorld{nullptr};
  const Renderer *m_shadedWith{nullptr};
  helium::TimeStamp m_shadedStamp{0};
  bool m_shadingDirty{true};

  float m_duration{0.f};
};

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Frame *, ANARI_FRAME);
}
