export const SUPPLIED_PIPELINE_ID = 'supplied-yolo-3stage-v1';
export const ACCEPTED_DIGIT_LENGTHS: ReadonlyArray<number> = [5, 6];

export type ReadingView = {
  digits: string | null;
  message: string;
  elapsedMs: number | null;
  region: {box: number[]; size: number[]} | null;
};

function isAcceptedReadingString(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    /^[0-9]+$/.test(value) &&
    ACCEPTED_DIGIT_LENGTHS.includes(value.length)
  );
}

function resolveMessage(digits: string | null, reason: unknown): string {
  if (digits !== null) {
    return 'Đối chiếu chỉ số và vùng nhận diện với đồng hồ trước khi sử dụng.';
  }
  switch (reason) {
    case 'no_row':
      return 'Chưa tìm thấy vùng số. Vui lòng chọn ảnh khác rõ và gần đồng hồ hơn.';
    case 'multiple_rows':
      return 'Có nhiều vùng số cạnh tranh. Vui lòng chọn ảnh chỉ có một đồng hồ.';
    case 'unusable_row_geometry':
      return 'Khung dãy số bị nghiêng hoặc biến dạng quá mức. Vui lòng chọn ảnh chụp rõ mặt đồng hồ hơn.';
    case 'no_characters':
      return 'Đã tìm thấy khung nhưng chưa nhận diện được chữ số. Vui lòng chọn ảnh rõ hơn, tránh lóa sáng.';
    case 'unexpected_length':
      return 'Số chữ số nhận diện được chưa đủ 5–6 chữ số theo chuẩn đồng hồ. Vui lòng chọn ảnh rõ toàn bộ dãy số.';
    case 'non_digit_output':
    case 'unsupported_label':
    case 'orientation_conflict':
      return 'Nhãn chữ số hoặc hướng khung chưa hợp lệ. Vui lòng chọn ảnh khác rõ mặt đồng hồ hơn.';
    default:
      return 'Chưa đọc chắc chắn. Vui lòng chọn ảnh khác rõ hơn, tránh chói mặt đồng hồ.';
  }
}

export function readingView(response: string): ReadingView {
  try {
    const value: unknown = JSON.parse(response);
    if (!value || typeof value !== 'object') throw new Error('Invalid OCR response');
    const result = value as Record<string, unknown>;
    const validPipeline =
      result.pipelineId === undefined || result.pipelineId === SUPPLIED_PIPELINE_ID;
    const digits =
      validPipeline && result.status === 'ok' && isAcceptedReadingString(result.rawDigits)
        ? result.rawDigits
        : null;
    const message = resolveMessage(digits, result.reason);
    const timing = result.totalMs ?? result.invokeMs;
    const elapsedMs =
      typeof timing === 'number' && Number.isFinite(timing) && timing >= 0 ? timing : null;
    const box = result.rowBox;
    const size = result.imageSize;
    const valid =
      Array.isArray(box) &&
      box.length === 4 &&
      Array.isArray(size) &&
      size.length === 2 &&
      [...box, ...size].every(v => typeof v === 'number' && Number.isFinite(v)) &&
      size[0] > 0 &&
      size[1] > 0 &&
      box[0] >= 0 &&
      box[1] >= 0 &&
      box[2] > box[0] &&
      box[3] > box[1] &&
      box[2] <= size[0] &&
      box[3] <= size[1];
    return {digits, message, elapsedMs, region: valid ? {box, size} : null};
  } catch {
    return {
      digits: null,
      message: 'Không đọc được phản hồi OCR. Hãy thử lại.',
      elapsedMs: null,
      region: null,
    };
  }
}
