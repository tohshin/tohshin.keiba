#!/usr/bin/env python3
"""ipatgo.exe 互換 CLI スクリプト
使用法:
  python ipatgo.py version
  python ipatgo.py stat <CardNo/InetID> <BirthDay/UserNo> <PassNo> <9999/ParsNo>
  python ipatgo.py file <CardNo/InetID> <BirthDay/UserNo> <PassNo> 9999 <投票ファイル> [--win5]
  python ipatgo.py data <InetID> <UserNo> <PassNo> <ParsNo> <投票データ行>
"""

import sys
import os

# プロジェクトルートをインポートパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ipat.cli import main

if __name__ == "__main__":
    sys.exit(main())
