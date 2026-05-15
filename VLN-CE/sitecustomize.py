try:
    import habitat
    import habitat.config
    import habitat.config.default as habitat_config_default
    from yacs.config import CfgNode

    if not hasattr(habitat, "Config"):
        habitat.Config = CfgNode
    if not hasattr(habitat.config, "Config"):
        habitat.config.Config = CfgNode
    if not hasattr(habitat_config_default, "Config"):
        habitat_config_default.Config = CfgNode
except Exception:
    pass

try:
    import cv2
    import numpy as np
    import habitat.utils.visualizations.utils as vis_utils

    if not hasattr(vis_utils, "append_text_to_image"):

        def append_text_to_image(image, text):
            image = np.asarray(image)
            if image.ndim != 3:
                return image

            pad = 70
            canvas = np.full(
                (image.shape[0] + pad, image.shape[1], image.shape[2]),
                255,
                dtype=image.dtype,
            )
            canvas[: image.shape[0]] = image
            words = str(text).split()
            lines = []
            current = ""
            for word in words:
                candidate = f"{current} {word}".strip()
                if len(candidate) > 90:
                    lines.append(current)
                    current = word
                else:
                    current = candidate
            if current:
                lines.append(current)
            for idx, line in enumerate(lines[:2]):
                cv2.putText(
                    canvas,
                    line,
                    (8, image.shape[0] + 24 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
            return canvas

        vis_utils.append_text_to_image = append_text_to_image
except Exception:
    pass

try:
    import habitat.core.utils as core_utils

    if not hasattr(core_utils, "try_cv2_import"):

        def try_cv2_import():
            import cv2

            return cv2

        core_utils.try_cv2_import = try_cv2_import
except Exception:
    pass
