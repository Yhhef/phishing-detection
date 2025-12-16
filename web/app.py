"""
Flask Web应用主程序
基于网络流量特征的钓鱼网站检测系统

作者: 毕业设计项目组
日期: 2025年12月
"""

# TODO: Day 21-28 实现Web系统

from flask import Flask

app = Flask(__name__)


@app.route('/')
def index():
    """首页"""
    return "钓鱼网站检测系统 - 开发中..."


if __name__ == '__main__':
    app.run(debug=True, port=5000)
