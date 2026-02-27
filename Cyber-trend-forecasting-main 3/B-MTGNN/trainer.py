import torch
from torch.nn.utils import clip_grad_norm_


class Optim:
    """
    Wrapper optimizer used in edit_train_test.py

    사용법 (edit_train_test.py 기준):
        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )

        ...
        grad_norm = optim.step()
    """

    def __init__(self, params, optim: str, lr: float, clip: float, lr_decay: float = None, weight_decay: float = 0.0):
        """
        params    : model.parameters()
        optim     : 'adam', 'sgd', 'rmsprop' 등
        lr        : learning rate
        clip      : gradient clipping max norm
        lr_decay  : 학습률 감소 비율 (None 이면 사용 안 함)
        """
        self.lr = lr
        self.clip = clip
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay

        optim = optim.lower()
        if optim == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=self.weight_decay)
        elif optim == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif optim == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optim}")

        # clipping 대상 파라미터들 저장
        self.params = list(self.optimizer.param_groups[0]["params"])

        # lr_decay가 0<gamma<1 범위일 때만 지수 감소 스케줄러 사용
        if lr_decay is not None and 0.0 < lr_decay < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=lr_decay
            )
        else:
            self.scheduler = None

    def step(self):
        """
        gradient clipping + optimizer.step() + (optional) lr scheduler.step()

        return: grad_norm (없으면 0.0)
        """
        grad_norm = 0.0

        # gradient clipping
        if self.clip is not None and self.clip > 0:
            grad_norm = float(clip_grad_norm_(self.params, self.clip))

        # 파라미터 업데이트
        self.optimizer.step()

        # lr decay 스케줄러가 있으면 한 스텝 진행
        if self.scheduler is not None:
            self.scheduler.step()

        return grad_norm
    