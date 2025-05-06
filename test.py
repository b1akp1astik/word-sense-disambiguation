from cs5322s25 import WSD_Test_camper, WSD_Test_conviction, WSD_Test_deed
from train import LemmaTransformer, GlossSimilarity, WindowFeatures, WordNetOverlap

# Example usage:
if __name__ == "__main__":
        print(WSD_Test_camper(["They parked their camper at the lakeside.", "Each camper brought a sleeping bag."])) # expected 1, 2
        print(WSD_Test_conviction(["He was convicted for murder.", "She spoke with great conviction."])) # expected 2, 1
        print(WSD_Test_deed(["He signed the deed to the property.", "Her brave deed saved a child."])) # 1, 2
