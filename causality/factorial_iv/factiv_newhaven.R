library(factiv)
library(dplyr)

data(newhaven)

summary(newhaven)

str(newhaven)

head(newhaven)

table(
  `In-Person` = newhaven$inperson_rand,
  `Phone` = newhaven$phone_rand
)

table(
  `Phone Assignment` = newhaven$phone_rand,
  `Phone Uptake` = newhaven$phone
)

out <- iv_finite_factorial(turnout_98 ~ inperson + phone | inperson_rand +
                             phone_rand, data = newhaven)
summary(out)

tidy(out)

out_sp <- iv_factorial(turnout_98 ~ inperson + phone | inperson_rand +
                         phone_rand, data = newhaven)
summary(out_sp)

tidy(out_sp, conf.int = TRUE)

cov_prof <- compliance_profile(
  ~ inperson + phone | inperson_rand + phone_rand |
    age + maj_party + turnout_96,
  data = newhaven)

cov_prof