from data import load_data
from trainer import train_flow, create_multiscale_flow, create_simple_flow

if __name__ == "__main__":
    from lightning.pytorch.callbacks import ModelSummary  # noqa

    train_loader, val_loader, test_loader = load_data()
    flow_dict = {"simple": {}, "vardeq": {}, "multiscale": {}}
    flow_dict["simple"]["model"], flow_dict["simple"]["result"] = train_flow(
        create_simple_flow(use_vardeq=False), train_loader, val_loader, test_loader, model_name="MNISTFlow_simple"
    )
    flow_dict["vardeq"]["model"], flow_dict["vardeq"]["result"] = train_flow(
        create_simple_flow(use_vardeq=True), train_loader, val_loader, test_loader, model_name="MNISTFlow_vardeq"
    )
    flow_dict["multiscale"]["model"], flow_dict["multiscale"]["result"] = train_flow(
        create_multiscale_flow(), train_loader, val_loader, test_loader, model_name="MNISTFlow_multiscale"
    )
