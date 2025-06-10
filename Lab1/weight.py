from model import ClassificationModel

if __name__ == "__main__":
    model = ClassificationModel()
    total_params = sum(p.numel() for p in model.parameters())
    print("# parameters:", total_params)
