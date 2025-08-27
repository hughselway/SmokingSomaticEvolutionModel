class SmokingRecord:
    status: str | None = None
    start_smoking_age: float | None = None
    stop_smoking_age: float | None = None
    pack_years: float | None = None

    def __init__(self, patient: str, age: float) -> None:
        self.patient = patient
        self.age = age

    def __str__(self) -> str:
        return (
            f'{self.status if self.status is not None else "Arbitrary"} record;'
            f"Patient ID: {self.patient}; "
            f"Age: {self.age}"
        )

    def smoking_at_age(self, current_age: float) -> bool:
        raise NotImplementedError(self, current_age)

    def get_intensity(self) -> float:
        """returns the smoking intensity in packs per day"""
        return (
            self.pack_years / self.get_smoking_duration()
            if self.pack_years is not None
            else 0
        )

    def get_smoking_duration(self) -> float:
        raise NotImplementedError(self)


class SmokerRecord(SmokingRecord):
    status: str = "smoker"

    def __init__(
        self, patient: str, age: float, start_smoking_age: float, pack_years: float
    ) -> None:
        super().__init__(patient, age)
        self.start_smoking_age = start_smoking_age
        self.pack_years = pack_years

    def __str__(self) -> str:
        return (
            f"{super().__str__()}; Age started smoking: {self.start_smoking_age}; "
            f"Pack years: {self.pack_years}"
        )

    def smoking_at_age(self, current_age: float) -> bool:
        assert self.start_smoking_age is not None
        return current_age >= self.start_smoking_age

    def get_smoking_duration(self) -> float:
        assert self.start_smoking_age is not None
        return self.age - self.start_smoking_age


class ExSmokerRecord(SmokingRecord):
    status: str = "ex-smoker"

    def __init__(
        self,
        patient: str,
        age: float,
        start_smoking_age: float,
        stop_smoking_age: float,
        pack_years: float,
    ) -> None:
        super().__init__(patient, age)
        self.start_smoking_age = start_smoking_age
        self.stop_smoking_age = stop_smoking_age
        self.pack_years = pack_years

    def __str__(self) -> str:
        return (
            f"{super().__str__()}; "
            f"Age started smoking: {self.start_smoking_age}; "
            f"Age stopped smoking: {self.stop_smoking_age}; "
            f"Pack years: {self.pack_years}"
        )

    def smoking_at_age(self, current_age: float) -> bool:
        assert self.start_smoking_age is not None and self.stop_smoking_age is not None
        return self.start_smoking_age <= current_age <= self.stop_smoking_age

    def get_smoking_duration(self) -> float:
        assert self.start_smoking_age is not None and self.stop_smoking_age is not None
        return self.stop_smoking_age - self.start_smoking_age


class NonSmokerRecord(SmokingRecord):
    status = "non-smoker"

    def smoking_at_age(self, current_age: float) -> bool:
        _ = self, current_age
        return False

    def get_smoking_duration(self) -> float:
        return 0.0
