# Utility functions

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")  # 先頭の # を削除
    # 6文字 (RRGGBB) でなければエラー
    if len(hex_color) != 6:
        raise ValueError("カラーコードは6桁の16進数で指定してください。")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)