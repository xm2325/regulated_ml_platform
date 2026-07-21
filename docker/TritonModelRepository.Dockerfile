FROM busybox:1.36.1
LABEL org.opencontainers.image.source="https://github.com/xm2325/regulated_ml_platform" \
      org.opencontainers.image.description="Immutable Triton model repository for the regulated AI reference platform" \
      org.opencontainers.image.licenses="MIT"
COPY models/triton/model_repository /model-repository
RUN test -s /model-repository/support_base/1/model.onnx \
    && test -s /model-repository/support_calibrator/1/model.onnx \
    && test -s /model-repository/support_ensemble/config.pbtxt \
    && test -s /model-repository/support_ensemble/1/version.txt
