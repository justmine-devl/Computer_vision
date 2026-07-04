def rearrange(x, pattern: str, **axes):
    pattern = " ".join(pattern.split())

    if pattern == "b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2":
        b, c, h, w = x.shape
        p1, p2 = axes["p1"], axes["p2"]
        return x.reshape(b, c, p1, h // p1, p2, w // p2)

    if pattern == "b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2":
        b, c, p1, w1, p2, w2 = x.shape
        return x.permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, c * p1 * p2, w1, w2)

    if pattern == "b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)":
        b, cp, w1, w2 = x.shape
        p1, p2 = axes["p1"], axes["p2"]
        c = cp // (p1 * p2)
        return x.reshape(b, c, p1, p2, w1, w2).permute(0, 1, 2, 4, 3, 5).contiguous().reshape(b, c, p1 * w1, p2 * w2)

    if pattern == "b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch":
        b, _, nw = x.shape
        nc, ch, owh, oww = axes["nc"], axes["ch"], axes["owh"], axes["oww"]
        return x.reshape(b, nc, ch, owh, oww, nw).permute(1, 0, 5, 3, 4, 2).contiguous().reshape(nc, b * nw, owh * oww, ch)

    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")
